import torch
import torch.nn as nn
import torch.nn.functional as F


class LiirNetwork(nn.Module):
    def __init__(self, input_shape, args):
        super(LiirNetwork, self).__init__()
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        self.args = args

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3_r_in = nn.Linear(args.critic_dim, self.n_actions)      # Intrinsic Reward
        self.fc3_v_mix = nn.Linear(args.critic_dim, 1)                  # Proxy Critic (Extrinsic and Intrinsic R)

        self.fc4 = nn.Linear(args.critic_dim * self.n_agents, 1)        # Extrinsic Critic

    def forward(self, inputs):                      # (bs, max_t, n_agents, feature_len) --> max_t = 1 or episode_limit
        x_1 = F.relu(self.fc1(inputs))              # (bs, max_t, n_agents, hidden_dim)
        x_1 = F.relu(self.fc2(x_1))
        v_mix = self.fc3_v_mix(x_1)                 # (bs, max_t, n_agents, 1)

        bs = inputs.shape[0]
        max_t = inputs.shape[1]

        x1 = x_1.reshape((bs, max_t, -1))           # (bs, max_t, n_agents * hidden_dim)
        v_ex = self.fc4(x1)                         # (bs, max_t, 1)
        x = self.fc3_r_in(x_1)                      # (bs, max_t, n_agents, n_actions)
        x1 = x / 10
        x2 = F.tanh(x1)
        r_in = x2 * 10

        return r_in, v_mix, v_ex
