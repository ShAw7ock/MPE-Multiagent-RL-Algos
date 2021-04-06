import torch
import os
from network.base_net import RNN
from network.coma_critic import ComaCritic


class COMA:
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

        self.eval_rnn = RNN(actor_input_shape, args)
        print('Init Algo Coma')

        self.eval_critic = ComaCritic(critic_input_shape, args)
        self.target_critic = ComaCritic(critic_input_shape, args)

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
        self.eval_hidden = None

    def learn(self, batch, max_episode_len, train_step, epsilon):
        bs = batch['o'].shape[0]
        self.init_hidden(bs)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, terminated = batch['u'], batch['r'], batch['terminated']
        if self.args.use_cuda:
            u = u.to(self.device)

        q_values = self._train_critic(batch, max_episode_len, train_step)
        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)

        q_taken = torch.gather(q_values, dim=3, index=u).squeeze(3)     # (bs, episode_limit, n_agents)
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)
        log_pi_taken = torch.log(pi_taken)

        # Advantage for actor(policy) optimization
        baseline = (q_values * action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
        advantage = (q_taken - baseline).detach()
        loss = - (advantage * log_pi_taken).mean()
        self.rnn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()

    def _train_critic(self, batch, max_episode_len, train_step):
        # Unlike the qmix or vdn which seems like q_learning to choose the argmax Q values as the q_targets
        # COMA is someway like the MADDPG or DDPG algorithm which is deterministic policy gradient method
        # So it requires the deterministic next action infos as 'u_next'
        u, r, terminated = batch['u'], batch['r'], batch['terminated']
        u_next = u[:, 1:]   # (bs, episode_limit - 1, n_agents, n_actions)
        padded_u_next = torch.zeros(*u[:, -1].shape, dtype=torch.long).unsqueeze(1)     # Add a step with zeros
        u_next = torch.cat((u_next, padded_u_next), dim=1)
        if self.args.use_cuda:
            u = u.to(self.device)
            u_next = u_next.to(self.device)
        q_evals, q_targets = self._get_q_values(batch, max_episode_len)     # (bs, episode_limit, n_agents, n_actions)
        q_values = q_evals.clone()  # As the returns of the function to calculate the advantage

        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)          # (bs, episode_limit, n_agents)
        q_targets = torch.gather(q_targets, dim=3, index=u_next).squeeze(3)
        targets = self._td_lambda_target(batch, max_episode_len, q_targets.cpu())
        if self.args.use_cuda:
            targets = targets.to(self.device)
        td_error = targets.detach() - q_evals

        loss = td_error.pow(2).mean()
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
        self.critic_optimizer.step()
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_critic.load_state_dict(self.eval_critic.state_dict())
        return q_values

    def _get_q_values(self, batch, max_episode_len):
        bs = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)
            if self.args.use_cuda:
                inputs.to(self.device)
                inputs_next.to(self.device)
            q_eval = self.eval_critic(inputs)                   # (bs * n_agents, n_actions)
            q_target = self.target_critic(inputs_next)

            q_eval = q_eval.view(bs, self.n_agents, -1)         # (bs, n_agents, n_actions)
            q_target = q_target.view(bs, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)

        q_evals = torch.stack(q_evals, dim=1)                   # (bs, episode_limit, n_agents, n_actions)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        obs, obs_next = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx]
        u_onehot = batch['u_onehot'][:, transition_idx]
        # The Q_targets require the next actions infos.
        # The last o_next has no actions info, so it is given the all-zero info
        if transition_idx != max_episode_len - 1:
            u_onehot_next = batch['u_onehot'][:, transition_idx + 1]
        else:
            u_onehot_next = torch.zeros(*u_onehot.shape)

        bs = obs.shape[0]
        # Here every agent requires the total action infos
        u_onehot = u_onehot.view((bs, 1, -1)).repeat(1, self.n_agents, 1)   # (bs, n_agents, n_agents * n_actions)
        u_onehot_next = u_onehot_next.view((bs, 1, -1)).repeat(1, self.n_agents, 1)

        """
        # If the last actions is required
        if transition_idx == 0:
            u_onehot_last = torch.zeros_like(u_onehot)
        else:
            u_onehot_last = batch['u_onehot'][:, transition_idx - 1]
            u_onehot_last = u_onehot_last.view((bs, 1, -1)).repeat(1, self.n_agents, 1)
        """

        inputs, inputs_next = [], []
        # States
        states = obs.view((bs, 1, -1)).repeat(1, self.n_agents, 1)              # (bs, n_agents, n_agents * obs_shape)
        states_next = obs_next.view((bs, 1, -1)).repeat(1, self.n_agents, 1)
        inputs.append(states)
        inputs_next.append(states_next)

        """
        # Last actions
        inputs.append(u_onehot_last)
        inputs_next.append(u_onehot)
        """

        # Observations
        inputs.append(obs)
        inputs_next.append(obs_next)

        # Current actions
        # COMA require the actions of other agents but not the agent itself, so its own action should be masked.
        action_mask = (1 - torch.eye(self.n_agents))
        action_mask = action_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)     # (n_agents, n_agents * n_actions)
        inputs.append(u_onehot * action_mask.unsqueeze(0))
        inputs_next.append(u_onehot_next * action_mask.unsqueeze(0))
        # Agent id
        inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(bs, -1, -1))
        inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(bs * self.n_agents, -1) for x in inputs_next], dim=1)

        return inputs, inputs_next

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

    def _get_critic_input_shape(self):
        # The coma critic network require the total state and total action infos
        # shape = ((o1,...,on), oi, ai, (a1,...,an)) --> note: mask out the agent i 's action with zeros
        # State (concatenation of all agents' local observations)
        input_shape = self.obs_shape * self.n_agents
        # Observation
        input_shape += self.obs_shape
        # agent_id
        input_shape += self.n_agents
        # Critic Network needs current action and last action infos (default without last actions)
        # Joint actions
        input_shape += self.n_actions * self.n_agents

        return input_shape

    def _td_lambda_target(self, batch, max_episode_len, q_targets):
        bs = batch['o'].shape[0]
        terminated = (1 - batch["terminated"].float()).repeat(1, 1, self.n_agents)
        r = batch['r'].repeat((1, 1, self.n_agents))    # (bs, episode_limit, 1) --> (bs, episode_limit, n_agents)

        n_step_returns = torch.zeros((bs, max_episode_len, self.n_agents, max_episode_len))
        for transition_idx in range(max_episode_len - 1, -1, -1):
            n_step_returns[:, transition_idx, :, 0] = r[:, transition_idx] + self.args.gamma *\
                                                      q_targets[:, transition_idx] * terminated[:, transition_idx]
            for n in range(1, max_episode_len - transition_idx):
                n_step_returns[:, transition_idx, :, n] = r[:, transition_idx] + self.args.gamma *\
                                                          n_step_returns[:, transition_idx + 1, :, n - 1]

        lambda_return = torch.zeros((bs, max_episode_len, self.n_agents))
        for transition_idx in range(max_episode_len):
            returns = torch.zeros((bs, self.n_agents))
            for n in range(1, max_episode_len - transition_idx):
                returns += pow(self.args.td_lambda, n - 1) * n_step_returns[:, transition_idx, :, n - 1]
                lambda_return[:, transition_idx] = (1 - self.args.td_lambda) * returns +\
                                                   pow(self.args.td_lambda, max_episode_len - transition_idx - 1) *\
                                                   n_step_returns[:, transition_idx, :, max_episode_len - transition_idx - 1]

        return lambda_return

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
