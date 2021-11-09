import torch
import torch.nn as nn
from itertools import chain


class ComaCritic(nn.Module):
    """
    The COMA critic network choose to establish a critic network for each agent. (without reused network)
    This file is using the raw infos of observation and action.
    """
    def __init__(self, args):
        super(ComaCritic, self).__init__()
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        self.hidden_dim = args.critic_hidden_dim
        critic_inputs_shape = self._get_critic_inputs_shape()

        self.critics = nn.ModuleList()

        # iterator over agents
        for _ in range(self.n_agents):
            idim = self.obs_shape + self.n_actions
            odim = self.n_actions

            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(critic_inputs_shape, self.hidden_dim))
            critic.add_module('critic_nl1', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(self.hidden_dim, self.hidden_dim))
            critic.add_module('critic_nl2', nn.LeakyReLU())
            critic.add_module('critic_fc3', nn.Linear(self.hidden_dim, odim))
            self.critics.append(critic)

        self.shared_modules = None

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        if self.shared_modules is not None:
            return chain(*[m.parameters() for m in self.shared_modules])
        else:
            pass

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        if self.shared_modules is not None:
            for p in self.shared_parameters():
                p.grad.data.mul_(1. / self.n_agents)
        else:
            pass

    def forward(self, inps, return_q=True, return_all_q=False):
        agents = range(self.n_agents)
        obs = [o for o, u in inps]                              # n_agents * (bs, episode_limit, obs_shape)
        u_onehot = [u for o, u in inps]
        inputs = [torch.cat((o, u), dim=2) for o, u in inps]    # n_agents * (bs, episode_limit, obs_shape + n_actions)

        all_other_sa = []
        for a_i in agents:
            sa = [inp for i, inp in enumerate(inputs) if i != a_i]
            all_other_sa.append(sa)

        # Calculate Q for each agent
        all_rets = []
        for i, a_i in enumerate(agents):
            agent_ret = []
            critic_in = torch.cat((obs[i], *all_other_sa[i]), dim=2)
            q_all_values = self.critics[a_i](critic_in)
            u = u_onehot[a_i].max(dim=2, keepdim=True)[1]
            q_values = torch.gather(q_all_values, dim=2, index=u)
            if return_q:
                agent_ret.append(q_values)
            if return_all_q:
                agent_ret.append(q_all_values)

            if len(agent_ret) == 1:
                all_rets.append(agent_ret[0])
            else:
                all_rets.append(agent_ret)

        return all_rets

    def _get_critic_inputs_shape(self):
        # Agent's local observation
        inputs_shape = self.obs_shape
        # Other agents' observations + actions
        inputs_shape += (self.obs_shape + self.n_actions) * (self.n_agents - 1)

        return inputs_shape

