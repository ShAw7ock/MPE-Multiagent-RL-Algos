import numpy as np
from torch import Tensor
from torch.autograd import Variable


class ReplayBuffer:
    def __init__(self, args):
        """
        Inputs:
            :param n_agents: (int) num of the multi agents to cooperate, note that not the compete
            :param obs_shape: (int) num of the agents' observation spaces, note that every agent is the same
            :param n_actions: (int) num of the agents' actions spaces, note that every agent is the same
            :param episode_limit: (int) num of the steps in each episode
            :param buffer_size: (int) max size of the replay buffer
        """
        self.n_agents = args.n_agents
        self.obs_shape = args.obs_shape
        self.n_actions = args.n_actions
        # self.state_shape = args.state_shape
        self.episode_limit = args.episode_limit
        self.size = args.buffer_size
        # Create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        'r': np.empty([self.size, self.episode_limit, 1]),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'terminated': np.empty([self.size, self.episode_limit, 1])
                        }

        self.current_size = 0  # index of first empty location in buffer (last index when full)
        self.current_idx = 0  # current index to write to (overwrite oldest data)

    def __len__(self):
        return self.current_size

    def push(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]
        idxs = self._get_storage_idx(inc=batch_size)
        # store the information
        self.buffers['o'][idxs] = episode_batch['o']
        self.buffers['u'][idxs] = episode_batch['u']
        self.buffers['r'][idxs] = episode_batch['r']
        self.buffers['o_next'][idxs] = episode_batch['o_next']
        self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
        self.buffers['terminated'][idxs] = episode_batch['terminated']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        temp_buffer['r'] = (temp_buffer['r'] - temp_buffer['r'].mean()) / temp_buffer['r'].std()
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

