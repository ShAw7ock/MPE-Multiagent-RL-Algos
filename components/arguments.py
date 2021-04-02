import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default='simple_spread', type=str,
                        help="Name of environment")
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument("--n_rollout_threads", default=1, type=int, help="For simple test, we assume here to be 1")
    parser.add_argument("--n_training_threads", default=4, type=int, help="While using the cpu")
    parser.add_argument("--episode_limit", default=25, type=int, help="MPE has no terminate in an episode")

    # The algorithm choices: vdn, qmix, coma, liir
    parser.add_argument('--algo', type=str, default='vdn', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=True,
                        help='whether to use the last action to choose action')
    parser.add_argument('--reuse_networks', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--use_cuda', type=bool, default=False)
    parser.add_argument('--display', type=bool, default=False, help='whether to render while training or evaluating')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    args = parser.parse_args()
    return args


def get_mixer_args(args):
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64  # Only if the two_hyper_layers == True
    args.lr = 5e-4

    # Epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch to train the agent
    args.n_episodes = 20000

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
