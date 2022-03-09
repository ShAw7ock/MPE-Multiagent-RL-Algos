import argparse
import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
from gym.spaces import Box, Tuple, Discrete
from torch.autograd import Variable
from pathlib import Path
from components.make_env import make_env
from components.env_wrappers import SubprocVecEnv, DummyVecEnv
from components.buffer import ReplayBuffer
from components.rollout import RolloutWorker
from components.arguments import get_common_args, get_mixer_args, get_coma_args, get_liir_args, get_maac_args
from agent.agent import Agents

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def make_parallel_env(env_id, n_parallel_envs, seed, discrete_action=True):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_parallel_envs == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_parallel_envs)])


def get_shape(sp):
    space, dim = 0, 0
    if isinstance(sp, Box):
        space = sp.shape[0]
        dim = sp.shape[0]
    elif isinstance(sp, Tuple):
        for p in sp.spaces:
            if isinstance(p, Box):
                space += p.shape[0]
                dim += p.shape[0]
            else:
                space += p.n
                dim += 1
    else:  # if the instance is 'Discrete', the action dim is 1
        space = sp.n
        dim = 1
    return space, dim


def get_env_scheme(env):
    """
    We assume that the environment is the default 'simple_spread'
    which has the same observation_space and action_space for each agent.
    """
    agent_init_params = {}

    """
    for acsp, obsp in zip(env.action_space, env.observation_space):
        observation_space, observation_dim = get_shape(obsp)
        action_space, action_dim = get_shape(acsp)
        agent_init_params.append({'observation_space': observation_space,
                                  'observation_dim': observation_dim,
                                  'action_space': action_space,
                                  'action_dim': action_dim})
    """

    num_agents = len(env.observation_space)
    observation_space, observation_dim = get_shape(env.observation_space[0])
    action_space, action_dim = get_shape(env.action_space[0])

    agent_init_params['n_agents'] = num_agents
    agent_init_params['observation_space'] = observation_space
    agent_init_params['observation_dim'] = observation_dim
    agent_init_params['action_space'] = action_space
    agent_init_params['action_dim'] = action_dim

    return agent_init_params


def runner(env, args):
    model_dir = Path('./models') / args.env_id / args.algo
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir()
                         if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)

    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    results_dir = run_dir / 'results'

    os.makedirs(str(log_dir))
    os.makedirs(str(results_dir))
    logger = SummaryWriter(str(log_dir))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.use_cuda and args.n_training_threads is not None:
        torch.set_num_threads(args.n_training_threads)

    agents = Agents(args)
    rolloutWorker = RolloutWorker(env, agents, args)
    buffer = ReplayBuffer(args)

    train_step = 0
    mean_episode_rewards = []

    for ep_i in range(0, args.n_episodes, args.n_parallel_envs):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + args.n_parallel_envs,
                                        args.n_episodes))
        if args.display:
            for env_show in env.envs:
                env_show.render('human')

        # Using the RolloutWork to interact with the environment (rollout the episodes >= 1)
        episodes, rews, mean_rews = [], [], []
        for episode_idx in range(args.n_rollouts):
            episode, ep_rew, mean_ep_rew = rolloutWorker.generate_episode(episode_idx)
            episodes.append(episode)
            rews.append(ep_rew)
            mean_rews.append(mean_ep_rew)
        episodes_batch = episodes[0]
        episodes.pop(0)
        for episode in episodes:
            for key in episodes_batch.keys():
                episodes_batch[key] = np.concatenate((episodes_batch[key], episode[key]), axis=0)
        buffer.push(episodes_batch)

        # Algorithms VDN and QMIX need the buffer but not the epsilon to train agents
        if args.algo.find('vdn') > -1 or args.algo.find('qmix') > -1:
            for _ in range(args.training_steps):
                mini_batch = buffer.sample(min(buffer.current_size, args.batch_size))
                agents.train(mini_batch, train_step)
                train_step += 1
        # Algorithms COMA, LIIR, MAAC needs the buffer and the epsilon to train agents
        else:
            for _ in range(args.training_steps):
                mini_batch = buffer.sample(min(buffer.current_size, args.batch_size))
                agents.train(mini_batch, train_step, rolloutWorker.epsilon)
                train_step += 1

        rews = np.mean(rews)
        mean_rews = np.mean(mean_rews)
        mean_episode_rewards.append(mean_rews)
        logger.add_scalar('mean_episode_rewards', mean_rews, ep_i)
        print("Episode {} : Total reward {} , Mean reward {}".format(ep_i + 1, rews, mean_rews))

        if ep_i % args.save_cycle < args.n_parallel_envs:
            os.makedirs(str(run_dir / 'incremental'), exist_ok=True)
            agents.save(str(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1))))
            agents.save(str(run_dir / 'model.pt'))

    agents.save(str(run_dir / 'model.pt'))
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()

    index = list(range(1, len(mean_episode_rewards) + 1))
    plt.plot(index, mean_episode_rewards)
    plt.ylabel("Mean Episode Rewards")
    plt.savefig(str(results_dir) + '/mean_episode_rewards.jpg')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    args = get_common_args()
    if args.algo.find('vdn') > -1 or args.algo.find('qmix') > -1:
        args = get_mixer_args(args)
    elif args.algo.find('coma') > -1 or args.algo.find('maddpg') > -1:
        args = get_coma_args(args)
    elif args.algo.find('maac') > -1:
        args = get_maac_args(args)
    else:
        args = get_liir_args(args)
    if args.env_id != "simple_spread":
        raise ValueError("We require the default environment SIMPLE_SPREAD ...")
    # env = make_env(args.env_id)
    env = make_parallel_env(args.env_id, args.n_parallel_envs, args.seed)
    scheme = get_env_scheme(env)
    args.n_agents = scheme['n_agents']
    args.obs_shape = scheme['observation_space']
    args.n_actions = scheme['action_space']

    runner(env, args)
