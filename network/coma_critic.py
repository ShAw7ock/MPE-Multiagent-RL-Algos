import torch
import torch.nn as nn


class ComaCritic(nn.Module):
    """
    The COMA critic network choose to establish a critic network for each agent. (without reused network)
    """
    def __init__(self, args):
        super(ComaCritic, self).__init__()
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        critic_inputs_shape = self._get_critic_inputs_shape()

        self.critics = nn.ModuleList()
        # iterator over agents
        for _ in range(self.n_agents):
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(critic_inputs_shape, args.critic_dim))
            critic.add_module('critic_nl1', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(args.critic_dim, args.critic_dim))
            critic.add_module('critic_nl2', nn.LeakyReLU())
            critic.add_module('critic_fc3', nn.Linear(args.critic_dim, self.n_actions))
            self.critics.append(critic)

    def forward(self, inps):
        agents = range(self.n_agents)
        obs = [o for o, u in inps]                                  # n_agents * (bs, episode_limit, obs_shape)
        inputs = [torch.cat((o, u), dim=2) for o, u in inps]        # n_agents * (bs, episode_limit, obs_shape + n_actions)
        bs = inputs[0].shape[0]
        max_t = inputs[0].shape[1]

        all_other_sa = []
        for a_i in agents:
            sa = [k for i, k in enumerate(inputs) if i != a_i]
            sa = torch.stack(sa, dim=2).reshape((bs, max_t, -1))    # (bs, episode_limit, (n_agents-1) * sa_shape)
            all_other_sa.append(sa)

        # Calculate Q for each agent
        agents_ret = []
        for i, a_i in enumerate(agents):
            critic_in = torch.cat((obs[i], all_other_sa[i]), dim=2)
            q_values = self.critics[a_i](critic_in)
            agents_ret.append(q_values)
        return agents_ret

    def _get_critic_inputs_shape(self):
        # Agent's local observation
        inputs_shape = self.obs_shape
        # Other agents' observations + actions
        inputs_shape += (self.obs_shape + self.n_actions) * (self.n_agents - 1)

        return inputs_shape
