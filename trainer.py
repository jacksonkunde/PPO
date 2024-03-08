"""
This file declares the PPOTrainer class, which is responsible for training a PPOAgent on a given environment,
and the PPOScheduler class, which is responsible for scheduling the learning rate decay.
"""
from tqdm import tqdm
import numpy as np
from numpy.random import Generator
import torch as t
from torch import Tensor
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import einops
from typing import List, Tuple, Literal, Union, Optional
from jaxtyping import Float, Int
import wandb
import gym
import time

# import from other files in this project
from agent import PPOAgent
from utils import PPOArgs, set_global_seeds, make_env
from memory import ReplayMinibatch

device = t.device("cuda" if t.cuda.is_available() else "cpu")

class PPOScheduler:
    def __init__(self, optimizer: Optimizer, initial_lr: float, end_lr: float, total_training_steps: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_training_steps = total_training_steps
        self.n_step_calls = 0

    def step(self):
        '''Implement linear learning rate decay so that after total_training_steps calls to step, the learning rate is end_lr.

        Do this by directly editing the learning rates inside each param group (i.e. `param_group["lr"] = ...`), for each param
        group in `self.optimizer.param_groups`.
        '''
        self.n_step_calls += 1
        frac = self.n_step_calls / self.total_training_steps
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)

#helper function
def make_optimizer(agent: PPOAgent, total_training_steps: int, initial_lr: float, end_lr: float) -> tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_training_steps)
    return (optimizer, scheduler)

### Helpers to calculate the total PPO objective function
def calc_clipped_surrogate_objective(
    probs: Categorical,
    mb_action: Int[Tensor, "minibatch_size *action_shape"],
    mb_advantages: Float[Tensor, "minibatch_size"],
    mb_logprobs: Float[Tensor, "minibatch_size"],
    clip_coef: float,
    eps: float = 1e-8
) -> Float[Tensor, ""]:
    '''Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs:
        a distribution containing the actor's unnormalized logits of shape (minibatch_size, num_actions)
    mb_action:
        what actions actions were taken in the sampled minibatch
    mb_advantages:
        advantages calculated from the sampled minibatch
    mb_logprobs:
        logprobs of the actions taken in the sampled minibatch (according to the old policy)
    clip_coef:
        amount of clipping, denoted by epsilon in Eq 7.
    eps:
        used to add to std dev of mb_advantages when normalizing (to avoid dividing by zero)
    '''
    assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape

    # get log probs from Categorical
    logp = probs.log_prob(mb_action)

    # subtract log probs (equivlent to subtraction in log space)
    logp_diff = logp - mb_logprobs

    p_diff = t.exp(logp_diff)

    # normalize the advantages
    norm_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)

    # exp to get probs
    non_clip_loss = p_diff * norm_advantages

    clip_loss = t.clip(p_diff, 1-clip_coef, 1+clip_coef) * norm_advantages

    return t.minimum(non_clip_loss, clip_loss).mean()

def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size"],
    mb_returns: Float[Tensor, "minibatch_size"],
    vf_coef: float
) -> Float[Tensor, ""]:
    '''Compute the value function portion of the loss function.

    values:
        the value function predictions for the sampled minibatch (using the updated critic network)
    mb_returns:
        the target for our updated critic network (computed as `advantages + values` from the old network)
    vf_coef:
        the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    assert values.shape == mb_returns.shape
    V_targ = mb_returns
    V_theta = values
    return vf_coef * (V_theta - V_targ).pow(2).mean()


def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''Return the entropy bonus term, suitable for gradient ascent.

    probs:
        the probability distribution for the current policy
    ent_coef:
        the coefficient for the entropy loss, which weights its contribution to the overall objective function. Denoted by c_2 in the paper.
    '''
    return ent_coef * probs.entropy().mean()

class PPOTrainer:

    def __init__(self, args: PPOArgs):
        set_global_seeds(args.seed)
        self.args = args
        self.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, self.run_name, args.mode) for i in range(args.num_envs)])
        self.agent = PPOAgent(self.args, self.envs).to(device)
        self.optimizer, self.scheduler = make_optimizer(self.agent, self.args.total_training_steps, self.args.learning_rate, 0.0)


    def rollout_phase(self) -> Optional[int]:
        '''
        This function populates the memory with a new set of experiences, using `self.agent.play_step`
        to step through the environment. It also returns the episode length of the most recently terminated
        episode (used in the progress bar readout).
        '''
        last_episode_len = None
        for step in range(self.args.num_steps):
            infos = self.agent.play_step()
            for info in infos:
                if "episode" in info.keys():
                    last_episode_len = info["episode"]["l"]
                    last_episode_return = info["episode"]["r"]
                    if self.args.use_wandb: wandb.log({
                        "episode_length": last_episode_len,
                        "episode_return": last_episode_return,
                    }, step=self.agent.step)
        return last_episode_len

    def learning_phase(self) -> None:
        '''
        This function does the following:

            - Generates minibatches from memory
            - Calculates the objective function, and takes an optimization step based on it
            - Clips the gradients (see detail #11)
            - Steps the learning rate scheduler
        '''
        minibatches = self.agent.get_minibatches()
        for minibatch in minibatches:
            objective = self.compute_ppo_objective(minibatch)
            objective.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def compute_ppo_objective(self, minibatch: ReplayMinibatch) -> Float[Tensor, ""]:
        '''
        Handles learning phase for a single minibatch. Returns objective function to be maximized.
        '''
        # give the actor the observations
        logits = self.agent.actor(minibatch.observations)
        m = Categorical(logits=logits)

        # compute the value of each state from the critic network
        values = self.agent.critic(minibatch.observations).squeeze()

        # compute objective fnc vals
        clipped_surrogate_objective = calc_clipped_surrogate_objective(m, minibatch.actions, minibatch.advantages, minibatch.logprobs, self.args.clip_coef)
        value_loss = calc_value_function_loss(values, minibatch.returns, self.args.vf_coef)
        entropy_bonus = calc_entropy_bonus(m, self.args.ent_coef)

        # as described in eq 9 of og paper
        total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus

        # log many things to w&b
        with t.inference_mode():
            newlogprob = m.log_prob(minibatch.actions)
            logratio = newlogprob - minibatch.logprobs
            ratio = logratio.exp()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        if self.args.use_wandb: wandb.log(dict(
            total_steps = self.agent.step,
            values = values.mean().item(),
            learning_rate = self.scheduler.optimizer.param_groups[0]["lr"],
            value_loss = value_loss.item(),
            clipped_surrogate_objective = clipped_surrogate_objective.item(),
            entropy = entropy_bonus.item(),
            approx_kl = approx_kl,
            clipfrac = np.mean(clipfracs)
        ), step=self.agent.step)

        return total_objective_function

    def train(self) -> None:

        if self.args.use_wandb: wandb.init(
            project=self.args.wandb_project_name,
            entity=self.args.wandb_entity,
            name=self.run_name,
            monitor_gym=self.args.capture_video
        )

        progress_bar = tqdm(range(self.args.total_phases))

        for epoch in progress_bar:

            last_episode_len = self.rollout_phase()
            if last_episode_len is not None:
                progress_bar.set_description(f"Epoch {epoch:02}, Episode length: {last_episode_len}")

            self.learning_phase()

        self.envs.close()
        if self.args.use_wandb:
            wandb.finish()