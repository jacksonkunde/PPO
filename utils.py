from dataclasses import dataclass
import torch as t
from typing import Literal, Optional
import numpy as np
import torch as t
import gym
from atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, ClipRewardEnv, ResizeObservation, GrayScaleObservation, FrameStack


@dataclass
class PPOArgs:

    # Basic / global
    seed: int = 1
    cuda: bool = t.cuda.is_available()
    env_id: str = "CartPole-v1"
    mode: Literal["classic-control", "atari", "mujoco"] = "classic-control"

    # Wandb / logging
    use_wandb: bool = False
    capture_video: bool = True
    exp_name: str = "PPO_Implementation"
    log_dir: str = "logs"
    wandb_project_name: str = "PPOCart"
    wandb_entity: str = None

    # Duration of different phases
    total_timesteps: int = 500000
    num_envs: int = 4
    num_steps: int = 128
    num_minibatches: int = 4
    batches_per_learning_phase: int = 4

    # Optimization hyperparameters
    learning_rate: float = 2.5e-4
    max_grad_norm: float = 0.5

    # Computing advantage function
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Computing other loss functions
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.25

    def __post_init__(self):
        self.batch_size = self.num_steps * self.num_envs
        assert self.batch_size % self.num_minibatches == 0, "batch_size must be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.total_phases = self.total_timesteps // self.batch_size
        self.total_training_steps = self.total_phases * self.batches_per_learning_phase * self.num_minibatches

def set_global_seeds(seed):
    '''Sets random seeds in several different ways (to guarantee reproducibility)
    '''
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    t.backends.cudnn.deterministic = True
    
def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str, mode: str = "classic-control", video_log_freq: Optional[int] = None):
    """Return a function that returns an environment after setting up boilerplate."""

    if video_log_freq is None:
        video_log_freq = {"classic-control": 40, "atari": 30}[mode]

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env, 
                    f"videos/{run_name}", 
                    episode_trigger = lambda x : x % video_log_freq == 0
                )
        if mode == "atari":
            env = prepare_atari_env(env)
        elif mode == "mujoco":
            env = prepare_mujoco_env(env)
        obs = env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return thunk

def prepare_atari_env(env: gym.Env):
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4)
    return env
