import numpy as np
import torch
from algos import VDN, QMIX, COMA, MADDPG, MAAC, LIIR
from torch.distributions import Categorical


class Agents:
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        if args.algo == 'vdn':
            self.policy = VDN(args)
        elif args.algo == 'qmix':
            self.policy = QMIX(args)
        elif args.algo == 'coma':
            self.policy = COMA(args)
        elif args.algo == 'maddpg':
            self.policy = MADDPG(args)
        elif args.algo == 'liir':
            self.policy = LIIR(args)
        elif args.algo == 'maac':
            self.policy = MAAC(args)
        else:
            raise Exception("No such algorithm")

        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.args = args
        print('Init Agents')

    def select_action(self, obs, last_action, agent_num, epsilon, evaluate=False):
        n_envs = self.args.n_parallel_envs
        inputs = obs.copy()
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1
        agent_id = np.expand_dims(agent_id, axis=0).repeat(n_envs, axis=0)

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_networks:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]
        inputs = torch.tensor(inputs, dtype=torch.float32)
        if self.args.use_cuda:
            inputs = inputs.to(self.device)
            hidden_state = hidden_state.to(self.device)

        # get q values
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
        # choose the action
        if self.args.algo == 'vdn' or self.args.algo == 'qmix':
            if np.random.uniform() < epsilon:
                actions = [np.random.choice(self.n_actions) for _ in range(n_envs)]
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
            else:
                actions = torch.argmax(q_value, dim=1).unsqueeze(1)     # shape=(n_envs, 1)
        else:
            actions = self._select_action_from_softmax(q_value.cpu(), epsilon, evaluate)
        actions_onehot = torch.zeros(n_envs, self.n_actions).scatter_(1, actions, 1)

        return actions, actions_onehot

    def _select_action_from_softmax(self, inputs, epsilon, evaluate=False):
        """
        :param inputs: q_values of all actions
        """
        avail_actions = torch.ones_like(inputs)  # (n, n_actions) --> all 1
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])

        prob = torch.nn.functional.softmax(inputs, dim=-1)
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob, dim=1)
        else:
            action = Categorical(prob).sample().long()
        return action.unsqueeze(1)

    def train(self, batch, train_step, epsilon=None):
        max_episode_len = self.args.episode_limit
        self.policy.learn(batch, max_episode_len, train_step, epsilon)

    def save(self, filename):
        params_dict = self.policy.get_params()
        torch.save(params_dict, filename)

    def load(self, filename):
        params_dict = torch.load(filename)
        self.policy.load_params(params_dict)
