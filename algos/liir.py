import copy
import os
import torch
from torch.optim import Adam
from network.base_net import RNN
from network.liir_critic import LiirNetwork


def build_td_lambda_targets(rewards, terminated, target_qs, r_in, v_ex, gamma, td_lambda):
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - torch.sum(terminated, dim=1))

    ret_ex = v_ex.new_zeros(*v_ex.shape)
    ret_ex[:, -1] = v_ex[:, -1] * (1 - torch.sum(terminated, dim=1))

    theta = 0.01
    rewards_broad = rewards.repeat(1, 1, r_in.shape[2])
    if len(r_in.shape) > 3:
        r_in = r_in.squeeze(-1)
    rewards_mix = rewards_broad + theta * r_in[:, :-1, :]

    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1, -1):
        ret_ex[:, t] = td_lambda * gamma * ret_ex[:, t + 1] + (
                    rewards[:, t] + (1 - td_lambda) * gamma * v_ex[:, t + 1] * (1 - terminated[:, t]))

        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + (
                    rewards_mix[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))

    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1], ret_ex[:, :-1]


class LIIR:
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape
        critic_input_shape = self._get_critic_input_shape()
        # Actor Network (RNN)
        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_networks:
            actor_input_shape += self.n_agents

        self.critic = LiirNetwork(critic_input_shape, args)
        self.target_critic = LiirNetwork(critic_input_shape, args)

        self.eval_rnn = RNN(actor_input_shape, args)
        self.target_rnn = RNN(actor_input_shape, args)
        print('Init Algo LIIR')

        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.eval_rnn = self.eval_rnn.to(self.device)
            self.target_rnn = self.target_rnn.to(self.device)
            self.critic = self.critic.to(self.device)
            self.target_critic = self.target_critic.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())

        self.agent_params = list(self.eval_rnn.parameters())
        self.critic_params = list(self.critic.fc1.parameters()) + list(self.critic.fc2.parameters()) + \
                             list(self.critic.fc3_v_mix.parameters())
        self.intrinsic_params = list(self.critic.fc3_r_in.parameters()) + list(self.critic.fc4.parameters())

        self.agent_optimizer = Adam(params=self.agent_params, lr=args.actor_lr)
        self.critic_optimizer = Adam(params=self.critic_params, lr=args.critic_lr)
        self.intrinsic_optimizer = Adam(params=self.intrinsic_params, lr=args.critic_lr)

        # The timing of updating the target networks is controlled by the LIIR itself ('train_step' is unnecessary)
        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.args = args
        self.eval_hidden = None
        self.target_hidden = None

    def learn(self, batch, max_episode_len, train_step, epsilon):
        bs = batch['o'].shape[0]
        max_t = batch['o'].shape[1]
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        rewards, actions, terminated = batch['r'][:, :-1], batch['u'][:, :], batch['terminated'][:, :-1]

        q_vals, target_mix, target_ex, v_ex, r_in = self._train_critic(batch, rewards, terminated, actions)

        actions = actions[:, :-1]                                               # (bs, max_t, n_agents, shape)

        # ------------------ Calculate policy grad --------------------------
        self.init_hidden(bs)
        mac_out = self._get_eval_action_prob(batch, max_t, epsilon)[:, :-1]     # (bs, max_t - 1, n_agents, n_actions)
        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)
        q_vals = q_vals.reshape(-1, 1)                                          # (bs * max_t - 1 * n_agents, 1)
        pi = mac_out.view(-1, self.n_actions)
        pi_taken = torch.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        log_pi_taken = torch.log(pi_taken)                                      # (bs * max_t - 1 * n_agents * 1)

        advantages = (target_mix.reshape(-1, 1) - q_vals).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        agent_loss = -(advantages * log_pi_taken).mean()
        self.agent_optimizer.zero_grad()
        agent_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimizer.step()

        # ---------------- Intrinsic loss Optimizer --------------------------
        v_ex_loss = ((v_ex - target_ex.detach()) ** 2).view(-1, 1).mean()

        # ------- pg1 --------
        self.init_hidden(bs)
        mac_out_old = self._get_target_action_prob(batch, max_t, epsilon)[:, :-1]
        mac_out_old = mac_out_old / mac_out_old.sum(dim=-1, keepdim=True)
        pi_old = mac_out_old.view(-1, self.n_actions)
        pi_old_taken = torch.gather(pi_old, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        log_pi_old_taken = torch.log(pi_old_taken)                              # (bs * max_t - 1 * n_agents * 1)
        log_pi_old_taken = log_pi_old_taken.reshape(-1, self.n_agents)          # (bs * max_t - 1, n_agents * 1)

        # ------- pg2 --------
        self.init_hidden(bs)
        mac_out_new = self._get_eval_action_prob(batch, max_t, epsilon)[:, :-1]
        mac_out_new = mac_out_new / mac_out_new.sum(dim=-1, keepdim=True)
        pi_new = mac_out_new.view(-1, self.n_actions)
        pi_new_taken = torch.gather(pi_new, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        log_pi_new_taken = torch.log(pi_new_taken)
        log_pi_new_taken = log_pi_new_taken.reshape(-1, self.n_agents)      # (bs * max_t - 1, n_agents * 1)
        neglogpac_new = - log_pi_new_taken.sum(-1)                          # (bs * max_t - 1, 1)

        pi2 = log_pi_taken.reshape(-1, self.n_agents).sum(-1).clone()
        ratio_new = torch.exp(- pi2 - neglogpac_new)

        adv_ex = (target_ex - v_ex.detach()).detach()
        adv_ex = (adv_ex - adv_ex.mean()) / (adv_ex.std() + 1e-8)

        # _______ gradient for pg 1 and 2---
        pg_loss1 = log_pi_old_taken.view(-1, 1).mean()
        pg_loss2 = (adv_ex.view(-1) * ratio_new).mean()
        self.target_rnn.zero_grad()
        pg_loss1_grad = torch.autograd.grad(pg_loss1, list(self.target_rnn.parameters()))

        self.eval_rnn.zero_grad()
        pg_loss2_grad = torch.autograd.grad(pg_loss2, list(self.eval_rnn.parameters()))

        total_grad = 0
        for grad1, grad2 in zip(pg_loss1_grad, pg_loss2_grad):
            total_grad += (grad1 * grad2).sum()

        target_mix = target_mix.reshape(-1, max_t - 1, self.n_agents)
        pg_ex_loss = (total_grad.detach() * target_mix).mean()

        intrinsic_loss = pg_ex_loss + v_ex_loss
        self.intrinsic_optimizer.zero_grad()
        intrinsic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.intrinsic_params, self.args.grad_norm_clip)
        self.intrinsic_optimizer.step()

        self._update_policy_old()

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_cycle >= 1.0:
            self._update_target()
            self.last_target_update_step = self.critic_training_steps

    def _train_critic(self, batch, rewards, terminated, actions):
        bs = batch['o'].shape[0]
        max_t = batch['o'].shape[1]
        inputs = self._get_critic_inputs(batch)
        if self.args.use_cuda:
            inputs.to(self.device)

        _, target_vals, target_val_ex = self.target_critic(inputs)
        r_in, _, target_val_ex_opt = self.critic(inputs)
        r_in_taken = torch.gather(r_in, dim=3, index=actions)
        r_in = r_in_taken.squeeze(-1)                                       # (bs, max_t, n_agents * 1)

        target_vals = target_vals.squeeze(-1)                               # (bs, max_t, n_agents * 1)

        # Here, <rewards> <terminated> --> (bs, max_t-1, 1)
        # And   <target_vals> <target_val_ex> <r_in> --> (bs, max_t, n_agents, 1)
        targets_mix, targets_ex = build_td_lambda_targets(rewards, terminated, target_vals, r_in, target_val_ex,
                                                          self.args.gamma, self.args.td_lambda)
        vals_mix = torch.zeros_like(target_vals)[:, :-1]
        vals_ex = target_val_ex_opt[:, :-1]

        for t in reversed(range(rewards.size(1))):
            inputs_t = self._get_critic_inputs(batch, transition_idx=t)     # (bs, 1, n_agents, shape)
            if self.args.use_cuda:
                inputs_t.to(self.device)
            _, q_t, _ = self.critic(inputs_t)                               # (bs, 1, n_agents, 1)
            vals_mix[:, t] = q_t.view(bs, self.n_agents)                    # (bs, n_agents * 1)
            targets_t = targets_mix[:, t]

            td_error_loss = (q_t.view(bs, self.n_agents) - targets_t.detach())
            loss = (td_error_loss ** 2).mean()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
            self.critic_training_steps += 1

        # Here, vals_mix, vals_ex --> (bs, max_t-1, n_agents * 1)
        # And,  targets_mix, targets_ex --> (bs, max_t-1, n_agents * 1)
        return vals_mix, targets_mix, targets_ex, vals_ex, r_in

    def _get_critic_inputs(self, batch, transition_idx=None):
        """
        :param transition_idx:
            if transition_idx is None, slice(None) makes the whole steps into the critic inputs
            if transition_idx is not None, ts represent the single time-step
        Because the LIIR Critic Network didn't choose the GRU Network, so the steps data could be used together
        """
        bs = batch['o'].shape[0]
        max_t = self.args.episode_limit if transition_idx is None else 1
        ts = slice(None) if transition_idx is None else slice(transition_idx, transition_idx + 1)
        obs, u_onehot = batch['o'][:, ts], batch['u_onehot'][:, ts]

        inputs = []
        # States
        states = obs.view((bs, max_t, 1, -1)).repeat(1, 1, self.n_agents, 1)
        inputs.append(states)
        # Actions (joint actions masked out by each agent)
        actions = u_onehot.view((bs, max_t, 1, -1)).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - torch.eye(self.n_agents))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))
        # Agent id
        inputs.append(torch.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = torch.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)

        return inputs

    def _get_actor_inputs(self, batch, transition_idx):
        # LIIR use the policy_new and policy_old to calculate the action probability respectively
        # So the policy doesn't need the obs_next
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

    def _get_eval_action_prob(self, batch, max_episode_len, epsilon):
        bs = batch['o'].shape[0]
        # The available actions for each agent. In MPE, an agent could choose every action at any time-step.
        avail_actions = torch.ones_like(batch['u_onehot'])  # (bs, episode_limit, n_agents, n_actions) --> all 1
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)
            if self.args.use_cuda:
                inputs = inputs.to(self.device)
                self.eval_hidden = self.eval_hidden.to(self.device)
            outputs, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # (bs * n_agents, n_actions)
            outputs = outputs.view(bs, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)

        action_prob = torch.stack(action_prob, dim=1).cpu()  # (bs, episode_limit, n_agents, n_actions)
        actions_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])
        action_prob = (1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / actions_num
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        if self.args.use_cuda:
            action_prob = action_prob.to(self.device)
        return action_prob

    def _get_target_action_prob(self, batch, max_episode_len, epsilon):
        bs = batch['o'].shape[0]
        # The available actions for each agent. In MPE, an agent could choose every action at any time-step.
        avail_actions = torch.ones_like(batch['u_onehot'])  # (bs, episode_limit, n_agents, n_actions) --> all 1
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)
            if self.args.use_cuda:
                inputs = inputs.to(self.device)
                self.target_hidden = self.eval_hidden.to(self.device)
            outputs, self.target_hidden = self.target_rnn(inputs, self.eval_hidden)  # (bs * n_agents, n_actions)
            outputs = outputs.view(bs, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)

        action_prob = torch.stack(action_prob, dim=1).cpu()  # (bs, episode_limit, n_agents, n_actions)
        actions_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])
        action_prob = (1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / actions_num
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        if self.args.use_cuda:
            action_prob = action_prob.to(self.device)
        return action_prob

    def _get_critic_input_shape(self):
        # State (concatenation of all agents' local observations)
        input_shape = self.obs_shape * self.n_agents
        # agent_id
        input_shape += self.n_agents
        # Critic Network needs current action and last action infos (default without last actions)
        # Joint actions (without last actions)
        input_shape += self.n_actions * self.n_agents

        return input_shape

    def init_hidden(self, batch_size):
        self.eval_hidden = torch.zeros((batch_size, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((batch_size, self.n_agents, self.args.rnn_hidden_dim))

    def _update_policy_old(self):
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())

    def _update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def get_params(self):
        return {'policy_new': self.eval_rnn.state_dict(),
                'critic': self.critic.state_dict()}

    def load_params(self, params_dict):
        self.eval_rnn.load_state_dict(params_dict['policy_new'])
        self.critic.load_state_dict(params_dict['critic'])
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
