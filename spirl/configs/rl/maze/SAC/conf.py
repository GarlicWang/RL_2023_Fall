from spirl.configs.rl.maze.base_conf import *
from spirl.rl.envs.maze import UMazeEnv

# configuration = {
#     'seed': 42,
#     'agent': MazeSACAgent,
#     'environment': UMazeEnv,
#     'data_dir': '.',
#     'num_epochs': 60,
#     'max_rollout_len': 300,
#     'n_steps_per_epoch': 10000,
#     'n_warmup_steps': 5e3,
#     # 'log_videos': True,
# }
# configuration = AttrDict(configuration)
configuration = {
    'seed': 42,
    'agent': MazeSACAgent,
    # 'environment': ACRandMaze0S40Env,
    'environment': UMazeEnv,
    'data_dir': '.',
    'num_epochs': 10,
    'max_rollout_len': 100,
    'n_steps_per_epoch': 1000,
    'n_warmup_steps': 3e3,
    # 'log_videos': True,
}
configuration = AttrDict(configuration)