import numpy as np
import torch
import copy


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        self.episode_limit = args.episode_limit
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        s, o, u, r, u_onehot, terminated = [], [], [], [], [], []
        obs = self.env.reset()
        # env.reset will return a LIST with shape (n_agents + 1)
        # including the local observation np.array for each agent and a total state np.array.
        if len(obs) > self.n_agents:
            state = obs[-1]
            obs = obs[:-1]
        terminate = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.n_agents, self.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminate and step < self.episode_limit:
            if self.args.display:
                for env_show in self.env.envs:
                    env_show.render('human')

            actions, actions_onehot = [], []
            # obs = obs.squeeze(0)
            for agent_num in range(self.n_agents):
                action = self.agents.select_action(obs[agent_num], last_action[agent_num], agent_num, epsilon)
                action_onehot = np.zeros(self.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                last_action[agent_num] = action_onehot

            # actions_onehot_env = [actions_onehot for _ in range(self.args.n_rollout_threads)]
            # obs_next, rewards, terminates, infos = self.env.step(actions_onehot_env)
            obs_next, rewards, terminates, infos = self.env.step(actions_onehot)
            state_next = infos["state"]
            rewards = np.array(rewards).reshape([-1, self.n_agents])
            reward = np.mean(rewards, axis=-1)
            terminates = np.array(terminates).reshape([-1, self.n_agents])
            terminate = np.mean(terminates, axis=-1)

            s.append(state)
            o.append(obs)
            u.append(np.reshape(actions, [self.n_agents, -1]))
            r.append(reward)
            u_onehot.append(actions_onehot)
            terminated.append(terminate)

            episode_reward += reward
            step += 1
            obs = obs_next
            state = state_next
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # Append the last infos
        # obs = obs.reshape([self.n_agents, -1])
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        o = o[:-1]
        s_next = s[1:]
        s = s[:-1]

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       o_next=o_next.copy(),
                       u_onehot=u_onehot.copy(),
                       s_next=s_next.copy(),
                       terminated=terminated.copy()
                       )
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        return episode, episode_reward, np.mean(r)
