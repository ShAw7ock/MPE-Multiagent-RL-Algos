import torch
import os
from network.base_net import RNN
from network.qmix_net import QMIXNet


class QMIX:
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        self.state_shape = args.state_shape
        input_shape = self.obs_shape
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_networks:
            input_shape += self.n_agents

        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)

        self.eval_qmix_net = QMIXNet(self.state_shape, args)
        self.target_qmix_net = QMIXNet(self.state_shape, args)

        self.args = args
        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.eval_rnn.to(self.device)
            self.target_rnn.to(self.device)
            self.eval_qmix_net.to(self.device)
            self.target_qmix_net.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        self.optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr)

        self.eval_hidden = None
        self.target_hidden = None
        print('Init algo QMIX')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        bs = batch['o'].shape[0]
        self.init_hidden(bs)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next = batch['s'], batch['s_next']
        o, o_next, u, r, terminated = batch['o'], batch['o_next'], batch['u'], batch['r'], batch['terminated']
        if self.args.use_cuda and torch.cuda.is_available():
            s = s.to(self.device)
            s_next = s_next.to(self.device)
            u = u.to(self.device)
            r = r.to(self.device)
            terminated = terminated.to(self.device)
        q_evals, q_targets = self._get_q_values(batch, max_episode_len)
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        q_targets = q_targets.max(dim=3)[0]

        # qmix algorithm needs the total states infos to calculate the mixer network distribution
        # In the MPE, envs don't support the original states infos due to the Complete Observable (Not the POMDP)
        # So the state can be seen as the concatenation of all the agents' observations
        states = s.reshape((bs, self.args.episode_limit, -1))
        states_next = s_next.reshape((bs, self.args.episode_limit, -1))
        q_total_eval = self.eval_qmix_net(q_evals, states)
        q_total_target = self.target_qmix_net(q_targets, states_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)
        td_error = targets.detach() - q_total_eval
        loss = td_error.pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def _get_q_values(self, batch, max_episode_len):
        bs = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)
            if self.args.use_cuda:
                inputs = inputs.to(self.device)
                inputs_next = inputs_next.to(self.device)
                self.eval_hidden = self.eval_hidden.to(self.device)
                self.target_hidden = self.target_hidden.to(self.device)
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            q_eval = q_eval.view(bs, self.n_agents, -1)
            q_target = q_target.view(bs, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)

        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def _get_inputs(self, batch, transition_idx):
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        bs = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])

        if self.args.reuse_networks:
            inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(bs, -1, -1))
            inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(bs * self.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def init_hidden(self, batch_size):
        self.eval_hidden = torch.zeros((batch_size, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((batch_size, self.n_agents, self.args.rnn_hidden_dim))

    def get_params(self):
        return {'eval_rnn': self.eval_rnn.state_dict(),
                'eval_qmix_net': self.eval_qmix_net.state_dict()}

    def load_params(self, params_dict):
        # Get parameters from save_dict
        self.eval_rnn.load_state_dict(params_dict['eval_rnn'])
        self.eval_qmix_net.load_state_dict(params_dict['eval_qmix_net'])
        # Copy the eval networks to target networks
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
