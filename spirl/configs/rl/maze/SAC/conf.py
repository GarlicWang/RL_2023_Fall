from spirl.configs.rl.maze.base_conf import *
from spirl.rl.envs.maze import UMazeEnv, MediumMazeEnv

wandb = AttrDict(
    wandb_project_name = 'SBTMDRL',
    wandb_entity_name='sbtmdrl',
    wandb_group='medium_maze',
)

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
    'environment': MediumMazeEnv,
    'data_dir': '.',
    'num_epochs': 50,
    'max_rollout_len': 1000,
    'n_steps_per_epoch': 10000,
    'n_warmup_steps': 5e3,
    # 'log_videos': True,
}
configuration = AttrDict(configuration)

