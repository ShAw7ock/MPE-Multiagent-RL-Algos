import torch
import torch.nn.functional as F
import os
from network.base_net import RNN
from network.maac_critic import MaacCritic


def disable_gradients(module):
    for p in module.parameters():
        p.requires_grad = False


def enable_gradients(module):
    for p in module.parameters():
        p.requires_grad = True


class MAAC:
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape
        # Actor Network (RNN)
        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_networks:
            actor_input_shape += self.n_agents

        self.eval_rnn = RNN(actor_input_shape, args)
        print('Init Algo MAAC')

        self.eval_critic = MaacCritic(args)
        self.target_critic = MaacCritic(args)

        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.eval_rnn.to(self.device)
            self.eval_critic.to(self.device)
            self.target_critic.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.target_critic.load_state_dict(self.eval_critic.state_dict())

        self.rnn_parameters = list(self.eval_rnn.parameters())
        self.critic_parameters = list(self.eval_critic.parameters())

        self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=args.critic_lr)
        self.rnn_optimizer = torch.optim.Adam(self.rnn_parameters, lr=args.actor_lr)

        self.args = args
        self.loss_func = torch.nn.MSELoss()
        self.eval_hidden = None

    def learn(self, batch, max_episode_len, train_step, epsilon):
        bs = batch['o'].shape[0]
        self.init_hidden(bs)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        u = batch['u']
        if self.args.use_cuda:
            u = u.to(self.device)

        critic_rets = self._train_critic(batch, train_step)
        q_taken, q_values = [], []
        for a_i, (q_eval, q_all, regs) in zip(range(self.n_agents), critic_rets):
            q_taken.append(q_eval)
            q_values.append(q_all)
        q_taken = torch.stack(q_taken, dim=2).squeeze(3)
        q_values = torch.stack(q_values, dim=2)
        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)
        log_pi_taken = torch.log(pi_taken)

        # Advantage for actor(policy) optimization
        baseline = (q_values * action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
        advantage = (q_taken - baseline).detach()
        loss = - (advantage * log_pi_taken).mean()

        self.rnn_optimizer.zero_grad()
        disable_gradients(self.eval_critic)
        loss.backward()
        enable_gradients(self.eval_critic)
        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()

    def _train_critic(self, batch, train_step):
        r, terminated = batch['r'], batch['terminated']
        if self.args.use_cuda:
            r = r.to(self.device)
            terminated = terminated.to(self.device)

        critic_in, target_critic_in = self._get_critic_inputs(batch)
        q_targets = self.target_critic(target_critic_in)
        critic_rets = self.eval_critic(critic_in, return_all_q=True, regularize=True)
        q_loss = 0
        for a_i, q_next, (q_eval, q_all, regs) in zip(range(self.n_agents), q_targets, critic_rets):
            target = r + self.args.gamma * q_next * (1 - terminated)
            q_loss += self.loss_func(target, q_eval)
            for reg in regs:
                q_loss += reg

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.eval_critic.scale_shared_grads()
        torch.nn.utils.clip_grad_norm_(self.eval_critic.parameters(), self.args.grad_norm_clip * self.n_agents)
        self.critic_optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_critic.load_state_dict(self.eval_critic.state_dict())

        return critic_rets

    def _get_action_prob(self, batch, max_episode_len, epsilon):
        bs = batch['o'].shape[0]
        # The available actions for each agent. In MPE, an agent could choose every action at any time-step.
        avail_actions = torch.ones_like(batch['u_onehot'])  # (bs, episode_limit, n_agents, n_actions) --> all 1
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)
            if self.args.use_cuda:
                inputs = inputs.to(self.device)
                self.eval_hidden = self.eval_hidden.to(self.device)
            outputs, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)     # outputs:(bs * n_agents, n_actions)
            outputs = outputs.view(bs, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)

        action_prob = torch.stack(action_prob, dim=1).cpu()     # (bs, episode_limit, n_agents, n_actions)
        actions_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])
        action_prob = (1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / actions_num
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        if self.args.use_cuda:
            action_prob = action_prob.to(self.device)
        return action_prob

    def _get_actor_inputs(self, batch, transition_idx):
        # Because the rnn agent actor network didn't initialize a target network, it requires none next infos
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        bs = obs.shape[0]
        # Observation
        inputs = [obs]

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_networks:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(bs, -1, -1))
        # Since the using of GRU network, the inputs shape should be shaped as 2 dimensions
        inputs = torch.cat([x.reshape(bs * self.args.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_critic_inputs(self, batch):
        """
        The MAAC algorithm handle the critic inputs with total steps (without transition_idx)
        """
        obs, obs_next = batch['o'], batch['o_next']     # (bs, episode_limit, n_agents, obs_shape)
        u_onehot = batch['u_onehot']                    # (bs, episode_limit, n_agents, n_actions)
        u_onehot_next = u_onehot[:, 1:]                 # (bs, episode_limit - 1, n_agents, n_actions)
        padded_next = torch.zeros(*u_onehot[:, -1].shape, dtype=torch.float32).unsqueeze(1)  # Add a step with zeros
        u_onehot_next = torch.cat((u_onehot_next, padded_next), dim=1)
        if self.args.use_cuda:
            obs = obs.to(self.device)
            obs_next = obs_next.to(self.device)
            u_onehot = u_onehot.to(self.device)
            u_onehot_next = u_onehot_next.to(self.device)

        agents_obs, agents_obs_next = [], []
        agents_u, agents_u_next = [], []
        for a_i in range(self.n_agents):
            agent_obs, agent_obs_next = obs[:, :, a_i], obs_next[:, :, a_i]         # (bs, episode_limit, obs_shape)
            agent_u, agent_u_next = u_onehot[:, :, a_i], u_onehot_next[:, :, a_i]   # (bs, episode_limit, n_actions)
            agents_obs.append(agent_obs)
            agents_obs_next.append(agent_obs_next)
            agents_u.append(agent_u)
            agents_u_next.append(agent_u_next)

        target_critic_in = list(zip(agents_obs_next, agents_u_next))
        critic_in = list(zip(agents_obs, agents_u))

        return critic_in, target_critic_in

    def init_hidden(self, batch_size):
        self.eval_hidden = torch.zeros((batch_size, self.n_agents, self.args.rnn_hidden_dim))

    def get_params(self):
        return {'eval_critic': self.eval_critic.state_dict(),
                'eval_rnn': self.eval_rnn.state_dict()}

    def load_params(self, params_dict):
        # Get parameters from save_dict
        self.eval_rnn.load_state_dict(params_dict['eval_rnn'])
        self.eval_critic.load_state_dict(params_dict['eval_critic'])
        # Copy the eval networks to target networks
        self.target_critic.load_state_dict(self.target_critic.state_dict())
