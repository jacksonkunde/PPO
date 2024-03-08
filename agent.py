from tqdm import tqdm
import numpy as np
from numpy.random import Generator
import torch as t
from torch import Tensor
from torch.optim.optimizer import Optimizer
import gym
import gym.envs.registration
from gym.envs.classic_control.cartpole import CartPoleEnv
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import einops
from pathlib import Path
from typing import List, Tuple, Literal, Union, Optional
from jaxtyping import Float, Int
import wandb

# import from other files in this project
from utils import PPOArgs
from memory import ReplayMemory

# define device
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer

def get_actor_and_critic_classic(num_obs: int, num_actions: int):
    '''
    Returns (actor, critic) in the "classic-control" case, according to diagram above.
    '''
    actor = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, num_actions), std=0.01)
    )
    critic = nn.Sequential(
        layer_init(nn.Linear(num_obs, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 1), std=0.01)
    )

    return (actor, critic)

def get_actor_and_critic_atari(obs_shape: Tuple[int], num_actions: int):
    '''
    Returns (actor, critic) in the "atari" case, according to diagram above.
    Assume that L = 8m + 4 for some m in Z
    '''
    L_after_convolutions = (obs_shape[-1] // 8) - 3
    in_features = 64 * L_after_convolutions * L_after_convolutions

    mid = nn.Sequential(
        layer_init(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, padding=0, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(in_channels=32, out_channels=64, padding=0, stride=2, kernel_size=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1, padding=0)),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(in_features, 512)),
        nn.ReLU()
    )

    actor = nn.Sequential(
        mid,
        layer_init(nn.Linear(512, num_actions), std=0.01)
    )

    critic = nn.Sequential(
        mid,
        layer_init(nn.Linear(512, 1), std=1)
    )

    return actor, critic

def get_actor_and_critic(
    envs: gym.vector.SyncVectorEnv,
    mode: Literal["classic-control", "atari"] = "classic-control",
) -> Tuple[nn.Module, nn.Module]:
    '''
    Returns (actor, critic), the networks used for PPO, in one of 3 different modes.
    '''
    assert mode in ["classic-control", "atari"]

    obs_shape = envs.single_observation_space.shape
    num_obs = np.array(obs_shape).prod()
    num_actions = (
        envs.single_action_space.n
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else np.array(envs.single_action_space.shape).prod()
    )

    if mode == "classic-control":
        actor, critic = get_actor_and_critic_classic(num_obs, num_actions)
    if mode == "atari":
        return get_actor_and_critic_atari(obs_shape, num_actions)

    return actor.to(device), critic.to(device)

class PPOAgent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, args: PPOArgs, envs: gym.vector.SyncVectorEnv):
        super().__init__()
        self.args = args
        self.envs = envs

        # Keep track of global number of steps taken by agent
        self.step = 0

        # Get actor and critic networks
        self.actor, self.critic = get_actor_and_critic(envs, mode=args.mode)

        # Define our first (obs, done), so we can start adding experiences to our replay memory
        obs = envs.reset()
        self.next_obs = t.tensor(obs).to(device, dtype=t.float)
        self.next_done = t.zeros(envs.num_envs).to(device, dtype=t.float)

        # Create our replay memory
        self.memory = ReplayMemory(args, envs)


    def play_step(self) -> List[dict]:
        '''
        Carries out a single interaction step between the agent and the environment, and adds results to the replay memory.

        Returns the list of info dicts returned from `self.envs.step`.
        '''
		# Get newest observations
        obs = self.next_obs
        dones = self.next_done

        with t.inference_mode():
          logits = self.actor(obs)
        m = Categorical(logits=logits)
        actions = m.sample()

        # step the env
        next_obs, rewards, next_dones, infos = self.envs.step(actions.cpu().numpy())

        logprobs = m.log_prob(actions)
        with t.inference_mode():
            values = self.critic(obs).flatten()

		    # Add to memory
        self.memory.add(obs, actions, logprobs, values, rewards, dones)

        # Set next observation, and increment global step counter
        self.next_obs = t.from_numpy(next_obs).to(device, dtype=t.float)
        self.next_done = t.from_numpy(next_dones).to(device, dtype=t.float)
        self.step += self.envs.num_envs

        # Return infos dict, for logging
        return infos


    def get_minibatches(self) -> None:
        '''
        Gets minibatches from the replay memory.
        '''
        with t.inference_mode():
            next_value = self.critic(self.next_obs).flatten()
        return self.memory.get_minibatches(next_value, self.next_done)
    