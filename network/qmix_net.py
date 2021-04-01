import torch.nn as nn
import torch
import torch.nn.functional as F


class QMIXNet(nn.Module):
    def __init__(self, state_shape, args):
        """
        :param state_shape: In particle env, the state space equals to the sum of all agents' observations
        """
        super(QMIXNet, self).__init__()
        self.state_shape = state_shape
        self.args = args
        # Since the q_values include multi-agents (n_agents * 1)
        # hyper_w1 needs to output a matrix, but pytorch can only output a vector(unlike the TensorFlow)
        # So we need to split the hyper_w1 to hyper_w1 and hyper_w2
        # hyper_w1 handle the dim 'n_agents' and hyper_w2 handle the hidden_dim infos
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(state_shape, args.n_agents * args.qmix_hidden_dim)

            self.hyper_w2 = nn.Linear(state_shape, args.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(state_shape, args.qmix_hidden_dim)

        self.hyper_b2 = nn.Sequential(nn.Linear(state_shape, args.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(args.qmix_hidden_dim, 1)
                                      )

    def forward(self, q_values, states):
        """
        :param q_values: (bs, episode_limit, n_agents)
        :param states: (bs, episode_limit, state_shape)
        """
        bs = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)     # (bs * episode_limit, 1, n_agents)
        states = states.reshape(-1, self.state_shape)           # (bs * episode_limit, state_shape)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)     # (bs * episode_limit, n_agents, hidden_dim)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)                      # (bs * episode_limit, 1, hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)                        # (bs * episode_limit, 1, hidden_dim)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)                      # (bs * episode_limit, hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)                                              # (bs * episode_limit, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2                                # (bs * episode_limit, 1, 1)
        q_total = q_total.view(bs, -1, 1)                                   # (bs, episode_limit, 1)

        return q_total
