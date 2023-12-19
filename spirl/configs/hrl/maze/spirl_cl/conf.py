import os
import copy

from spirl.utils.general_utils import AttrDict
from spirl.rl.components.agent import FixedIntervalHierarchicalAgent
from spirl.rl.components.critic import SplitObsMLPCritic
from spirl.rl.components.sampler import ACMultiImageAugmentedHierarchicalSampler
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from spirl.rl.envs.maze import ACRandMaze0S40Env
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.policies.cl_model_policies import ACClModelPolicy
from spirl.data.maze.src.maze_agents import MazeACActionPriorSACAgent
from spirl.models.closed_loop_spirl_mdl import ImageClSPiRLMdl
from spirl.configs.default_data_configs.maze import data_spec


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'hierarchical RL on the maze env'

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': ACRandMaze0S40Env,
    'sampler': ACMultiImageAugmentedHierarchicalSampler,
    'data_dir': '.',
    'num_epochs': 60,
    'max_rollout_len': 2000,
    'n_steps_per_epoch': 1e4,
    'n_warmup_steps': 5e3,
}
configuration = AttrDict(configuration)


# Replay Buffer
replay_params = AttrDict(
    capacity=1e5,
    dump_replay=False,
)

# Observation Normalization
obs_norm_params = AttrDict(
)

sampler_config = AttrDict(
    n_frames=2,
)

base_agent_params = AttrDict(
    batch_size=256,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
)


###### Low-Level ######
# LL Policy Model
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=1e-2,
    prior_input_res=data_spec.res,
    n_input_frames=2,
    cond_decode=True,
)

# LL Policy
ll_policy_params = AttrDict(
    policy_model=ImageClSPiRLMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/maze/hierarchical_cl"),
    initial_log_sigma=-50.,
)
ll_policy_params.update(ll_model_params)

# LL Critic
ll_critic_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=data_spec.state_dim,
    output_dim=1,
    action_input=True,
    unused_obs_size=10,     # ignore HL policy z output in observation for LL critic
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    policy=ACClModelPolicy,
    policy_params=ll_policy_params,
    critic=SplitObsMLPCritic,
    critic_params=ll_critic_params,
))


###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=10,       # z-dimension of the skill VAE
    input_dim=data_spec.state_dim,
    max_action_range=2.,        # prior is Gaussian with unit variance
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=hl_policy_params.input_dim,
    output_dim=1,
    n_layers=2,  # number of policy network layers
    nz_mid=256,
    action_input=True,
    unused_obs_size=ll_model_params.prior_input_res **2 * 3 * ll_model_params.n_input_frames,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=ACLearnedPriorAugmentedPIPolicy,
    policy_params=hl_policy_params,
    critic=SplitObsMLPCritic,
    critic_params=hl_critic_params,
    td_schedule_params=AttrDict(p=1.),
))


##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=MazeACActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_videos=True,
    update_hl=True,
    update_ll=False,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    reward_norm=1.,
    screen_height=ll_model_params.prior_input_res,
    screen_width=ll_model_params.prior_input_res,
)

