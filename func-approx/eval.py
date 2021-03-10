from ray.rllib.rollout import create_parser, run

# The imports are necessary since the configs register the model and the environment
# noinspection PyUnresolvedReferences
import configs.gridworld_sa

# noinspection PyUnresolvedReferences
import configs.gridworld_ma

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
