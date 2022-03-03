import numpy as np


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        self.episode_limit = args.episode_limit
        self.n_envs = args.n_parallel_envs
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, u_onehot, terminated = [], [], [], [], []
        obs = self.env.reset()
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.n_agents, self.n_envs, self.n_actions))
        self.agents.policy.init_hidden(self.n_envs)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # Particle Envs has no terminal state, use limited steps instead.
        while step < self.episode_limit:
            if self.args.display:
                for env_show in self.env.envs:
                    env_show.render('human')

            actions, actions_onehot = [], []
            for agent_num in range(self.n_agents):
                action, action_onehot = self.agents.select_action(obs[:, agent_num],
                                                                  last_action[agent_num], agent_num, epsilon)
                actions.append(action.data.numpy())
                actions_onehot.append(action_onehot.data.numpy())
                last_action[agent_num] = action_onehot

            # shape=(n_agents, n_envs, -1) --> shape=(n_envs, n_agents, -1)
            actions_onehot_env = [[ac[i] for ac in actions_onehot] for i in range(self.n_envs)]
            actions_env = [[ac[i] for ac in actions] for i in range(self.n_envs)]
            # Step
            obs_next, rewards, terminates, infos = self.env.step(actions_onehot_env)
            rewards = np.array(rewards).reshape([-1, self.n_agents])
            reward = np.mean(rewards, axis=1)
            terminates = np.array(terminates).reshape([-1, self.n_agents])
            terminate = np.mean(terminates, axis=1)

            o.append(obs)                                                           # shape=(n_env, n_agents, -1)
            u.append(actions_env)                                                   # shape=(n_env, n_agents, 1)
            r.append(reward.reshape([self.n_envs, 1]))                              # shape=(n_env, 1)
            u_onehot.append(actions_onehot_env)                                     # shape=(n_env, n_agents, -1)
            terminated.append(terminate.reshape([self.n_envs, 1]))                  # shape=(n_env, 1)

            episode_reward += np.mean(reward)
            step += 1
            obs = obs_next
            # state = state_next
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # Append the last infos
        o.append(obs)
        o_next = o[1:]
        o = o[:-1]

        episode = dict(o=o.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       o_next=o_next.copy(),
                       u_onehot=u_onehot.copy(),
                       terminated=terminated.copy()
                       )
        for key in episode.keys():
            if key == 'r' or key == 'terminated':
                episode[key] = np.array(episode[key]).transpose((1, 0, 2))
            else:
                episode[key] = np.array(episode[key]).transpose((1, 0, 2, 3))
        if not evaluate:
            self.epsilon = epsilon

        return episode, episode_reward, np.mean(r)
