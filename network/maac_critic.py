import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain


class MaacCritic(nn.Module):
    """
    Attention network, used as critic for all agents.
    Each agent gets its own observation and action,
    and can also attend over the other agents' encoded observations and actions.
    """
    def __init__(self, args):
        super(MaacCritic, self).__init__()
        self.hidden_dim = args.critic_hidden_dim
        self.attend_heads = args.attend_heads
        assert (self.hidden_dim % self.attend_heads) == 0
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterator over agents
        for _ in range(self.n_agents):
            idim = self.obs_shape + self.n_actions
            odim = self.n_actions
            encoder = nn.Sequential()
            # TODO: the normalization needs the dimension to be (bs * episode_limit, shape) --> default with False
            if args.norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim, affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, self.hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * self.hidden_dim, self.hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(self.hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if args.norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(self.obs_shape, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(self.obs_shape, self.hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = self.hidden_dim // self.attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(self.attend_heads):
            self.key_extractors.append(nn.Linear(self.hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(self.hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(self.hidden_dim, attend_dim), nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

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

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for regularization
            return_attend (bool): return attention weights per agent
        """
        if agents is None:
            agents = range(self.n_agents)
        obs = [o for o, u in inps]                                  # n_agents * (bs, episode_limit, obs_shape)
        u_onehot = [u for o, u in inps]                             # n_agents * (bs, episode_limit, n_actions)
        inputs = [torch.cat((o, u), dim=2) for o, u in inps]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inputs)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](obs[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per attend_head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            # values don't include the information of the agent itself (baseline of COMA)
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]        # (n_agents-1) * (bs, episode_limit, dim)
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]    # (n_agents-1) * (bs, episode_limit, dim)
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], selector.shape[1], 1, -1),
                                             torch.stack(keys).permute(1, 2, 3, 0))     # (bs, episode_limit, 1, n_agents-1)
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[2])
                attend_weights = F.softmax(scaled_attend_logits, dim=3)
                other_values = (torch.stack(values).permute(1, 2, 3, 0) * attend_weights).sum(dim=3)    # (bs, episode_limit, dim)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)

        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=2)
            all_q = self.critics[a_i](critic_in)                                    # (bs, episode_limit, n_actions)
            int_acs = u_onehot[a_i].max(dim=2, keepdim=True)[1]
            q = torch.gather(all_q, dim=2, index=int_acs)                           # (bs, episode_limit, 1)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))

            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)

        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
