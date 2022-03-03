import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default='simple_spread', type=str,
                        help="Name of environment")
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument("--n_parallel_envs", default=5, type=int, help="The parallel envs to create")
    parser.add_argument("--n_training_threads", default=12, type=int, help="While using the cpu")
    parser.add_argument("--episode_limit", default=50, type=int, help="MPE has no terminate in an episode")

    # The algorithm choices: vdn, qmix, coma, liir, maac
    parser.add_argument('--algo', type=str, required=True, help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=False,
                        help='whether to use the last action to choose action')
    parser.add_argument('--reuse_networks', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--use_cuda', type=bool, default=False)
    parser.add_argument('--display', type=bool, default=False, help='whether to render while training or evaluating')
    parser.add_argument('--run_num', type=int, default=None, help='if evaluation mode, specify the model-saved file')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    args = parser.parse_args()
    return args


def get_mixer_args(args):
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64  # Only if the two_hyper_layers == True
    args.lr = 1e-3

    # Epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 150000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch to train the agent
    args.n_episodes = 200000

    # the number of sampling data episode
    args.n_rollouts = 1

    # the number of training steps in one episode
    args.training_steps = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 10000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.embedding_dim = 32
    args.critic_hidden_dim = 128
    args.actor_lr = 1e-3
    args.critic_lr = 1e-3
    args.norm_in = False

    # epsilon-greedy
    args.epsilon = 0.5
    args.min_epsilon = 0.01
    anneal_steps = 10000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'epoch'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # the number of the epoch to train the agent
    args.n_episodes = 200000

    # the number of training steps in one episode
    args.training_steps = 1

    # the number of the episodes in one epoch
    args.n_rollouts = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 20000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


def get_liir_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 256
    args.actor_lr = 1e-3
    args.critic_lr = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.min_epsilon = 0.01
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'epoch'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # the number of the epoch to train the agent
    args.n_episodes = 200000

    # the number of training steps in one episode
    args.training_steps = 1

    # the number of the episodes in one epoch
    args.n_rollouts = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 10000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


def get_maac_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_hidden_dim = 128
    args.actor_lr = 1e-3
    args.critic_lr = 1e-3
    args.attend_heads = 4
    args.norm_in = False

    # epsilon-greedy
    args.epsilon = 0.5
    args.min_epsilon = 0.01
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_episodes = 200000

    # the number of training steps in one episode
    args.training_steps = 1

    # the number of the episodes in one epoch
    args.n_rollouts = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 10000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args
