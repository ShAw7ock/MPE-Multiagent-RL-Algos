import torch
import torch.nn as nn
from itertools import chain


class ComaCritic(nn.Module):
    """
    The COMA critic network choose to establish a critic network for each agent. (without reused network)
    This file is using the encoding network to encode the observation and action infos.
    """
    def __init__(self, args):
        super(ComaCritic, self).__init__()
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        self.hidden_dim = args.critic_hidden_dim
        self.embedding_dim = args.embedding_dim

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterator over agents
        for _ in range(self.n_agents):
            idim = self.obs_shape + self.n_actions
            odim = self.n_actions
            encoder = nn.Sequential()
            if args.norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim, affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, self.embedding_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)

            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(self.n_agents * self.embedding_dim, self.hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(self.hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if args.norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(self.obs_shape, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(self.obs_shape, self.embedding_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        self.shared_modules = [self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.n_agents)

    def forward(self, inps, return_q=True, return_all_q=False):
        agents = range(self.n_agents)
        obs = [o for o, u in inps]                                  # n_agents * (bs, episode_limit, obs_shape)
        u_onehot = [u for o, u in inps]
        inputs = [torch.cat((o, u), dim=2) for o, u in inps]        # n_agents * (bs, episode_limit, obs_shape + n_actions)
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inputs)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](obs[a_i]) for a_i in agents]

        all_other_sa = []
        for a_i in agents:
            sa = [sa_encoding for i, sa_encoding in enumerate(sa_encodings) if i != a_i]
            all_other_sa.append(sa)

        # Calculate Q for each agent
        all_rets = []
        for i, a_i in enumerate(agents):
            agent_ret = []
            critic_in = torch.cat((s_encodings[i], *all_other_sa[i]), dim=2)
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
