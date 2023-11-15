# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Deep Policy-based Reinforcement Learning
#
# <img src="https://raw.githubusercontent.com/jeremiedecock/polytechnique-inf639-2023-students/master/logo.jpg" style="float: left; width: 15%" />
#
# [INF639-2023](https://moodle.polytechnique.fr/course/view.php?id=17866) Lab session #2
#
# 2019-2023 Jérémie Decock

# %% [markdown]
# [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeremiedecock/polytechnique-inf639-2023-students/blob/master/lab2_deep_policy-based_reinforcement_learning.ipynb)
#
# [![My Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jeremiedecock/polytechnique-inf639-2023-students/master?filepath=lab2_deep_policy-based_reinforcement_learning.ipynb)
#
# [![NbViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/jeremiedecock/polytechnique-inf639-2023-students/blob/master/lab2_deep_policy-based_reinforcement_learning.ipynb)
#
# [![Local](https://img.shields.io/badge/Local-Save%20As...-blue)](https://github.com/jeremiedecock/polytechnique-inf639-2023-students/raw/master/lab2_deep_policy-based_reinforcement_learning.ipynb)

# %% [markdown]
# ## Introduction
#
# The aim of this lab is to provide an in-depth exploration of policy-based reinforcement learning techniques, with a particular focus on the *Monte Carlo Policy Gradient (REINFORCE)* and *Actor Critic* methods.
#
# In this Python notebook, you'll have the opportunity to implement and assess several renowned Policy Gradient techniques.
#
# You can either:
# - open, edit and execute the notebook in *Google Colab* following this link: https://colab.research.google.com/github/jeremiedecock/polytechnique-inf639-2023-students/blob/master/lab2_deep_policy-based_reinforcement_learning.ipynb ; this is the **recommended** choice as you have nothing to install on your computer
# - open, edit and execute the notebook in *MyBinder* (if for any reason the Google Colab solution doesn't work): https://mybinder.org/v2/gh/jeremiedecock/polytechnique-inf639-2023-students/master?filepath=lab2_deep_policy-based_reinforcement_learning.ipynb
# - download, edit and execute the notebook on your computer if Python3 and JypyterLab are already installed: https://github.com/jeremiedecock/polytechnique-inf639-2023-students/raw/master/lab2_deep_policy-based_reinforcement_learning.ipynb
#
# If you work with Google Colab or MyBinder, **remember to save or download your work regularly or you may lose it!**

# %% [markdown]
# ## Setup the Python environment

# %% [markdown]
# ### Install required libraries

# %% [markdown]
# **Note**: This notebook relies on several libraries including `PyTorch`, `Gymnasium`, `NumPy`, `Pandas`, `Seaborn`, `imageio`, `pygame`, and `tqdm`.
# A complete list of dependencies can be found in the provided [requirements.txt](https://raw.githubusercontent.com/jeremiedecock/polytechnique-inf639-2023-students/master/requirements.txt) file.
#
# If you are using Google Colab, the uncomment and execute the next cell to install dependencies.

# %%
# #! pip install -r requirements.txt

# %% [markdown]
# If you are running this notebook on your local machine, download the [requirements.txt](https://raw.githubusercontent.com/jeremiedecock/polytechnique-inf639-2023-students/master/requirements.txt) file and place it in the same directory as this notebook. Then, execute the following command:
#
# ```
# pip install -r requirements.txt
# ```

# %% [markdown]
# ### Import required packages

# %%
# %matplotlib inline

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import random
import seaborn as sns
import torch
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.notebook import tqdm
from typing import List, Tuple, Union

from inf581 import *

from IPython.display import Image   # To display GIF images in the notebook

# %%
sns.set_context("talk")

# %%
# Set the device to CUDA if available, otherwise use CPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")


# %% [markdown]
# ## Part 1: Monte Carlo Policy Gradient (REINFORCE)

# %% [markdown]
# ### The Policy Gradient theorem

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# We will solve the CartPole environment using a policy gradient method which directly searchs in a family of parameterized policies $\pi_\theta$ for the optimal policy.
#
# This method performs gradient ascent in the policy space so that the total return is maximized.
# We will restrict our work to episodic tasks, *i.e.* tasks that have a starting states and last for a finite and fixed number of steps $T$, called horizon. 
#
# More formally, we define an optimization criterion that we want to maximize:
#
# $$J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=1}^T r(s_t,a_t)\right],$$
#
# where $\mathbb{E}_{\pi_\theta}$ means $a \sim \pi_\theta(s,.)$ and $T$ is the horizon of the episode.
# In other words, we want to maximize the value of the starting state: $V^{\pi_\theta}(s)$.
# The policy gradient theorem tells us that:
#
# $$
# \nabla_\theta J(\theta) = \nabla_\theta V^{\pi_\theta}(s) = \mathbb{E}_{\pi_\theta} \left[\nabla_\theta \log \pi_\theta (s,a) ~ Q^{\pi_\theta}(s,a) \right],
# $$
#
# where the $Q$-function is defined as:
#
# $$Q^{\pi_\theta}(s,a) = \mathbb{E}^{\pi_\theta} \left[\sum_{t=1}^T r(s_t,a_t)|s=s_1, a=a_1\right].$$
#
# The policy gradient theorem is particularly effective because it allows gradient computation without needing to understand the system's dynamics, as long as the $Q$-function for the current policy is computable. By simply applying the policy and observing the one-step transitions, sufficient information is gathered. Implementing a stochastic gradient ascent and substituting $Q^{\pi_\theta}(s_t,a_t)$ with a Monte Carlo estimate $R_t = \sum_{t'=t}^T r(s_{t'},a_{t'})$ for a single trajectory, we derive the REINFORCE algorithm.

# %% [markdown]
# The REINFORCE algorithm, introduced by Williams in 1992, is a Monte Carlo policy gradient method. It updates the policy in the direction that maximizes rewards, using full-episode returns as an unbiased estimate of the gradient. Each step involves generating an episode using the current policy, computing the gradient estimate, and updating the policy parameters. This algorithm is simple yet powerful, and it's particularly effective in environments where the policy gradient is noisy or the dynamics are complex.
#
# For further reading and a deeper understanding, refer to Williams' seminal paper (https://link.springer.com/article/10.1007/BF00992696) and the comprehensive text on reinforcement learning by Richard S. Sutton and Andrew G. Barto: "Reinforcement Learning: An Introduction", chap.13 (http://incompleteideas.net/book/RLbook2020.pdf).

# %% [markdown]
# Here is the REINFORCE algorithm.

# %% [markdown]
# ### Monte Carlo policy gradient (REINFORCE)
#
# <b>REQUIRE</b> <br>
#  $\quad$ A differentiable policy $\pi_{\boldsymbol{\theta}}$ <br>
#  $\quad$ A learning rate $\alpha \in \mathbb{R}^+$ <br>
# <b>INITIALIZATION</b> <br>
#  $\quad$ Initialize parameters $\boldsymbol{\theta} \in \mathbb{R}^d$ <br>
# <br>
# <b>FOR EACH</b> episode <br>
#     $\quad$ Generate full trace $\tau = \{ \boldsymbol{s}_0, \boldsymbol{a}_0, r_1, \boldsymbol{s}_1, \boldsymbol{a}_1, \dots, r_T, \boldsymbol{s}_T \}$ following $\pi_{\boldsymbol{\theta}}$ <br>
#     $\quad$ <b>FOR</b> $~ t=0,\dots,T-1$ <br>
#         $\quad\quad$ $G \leftarrow \sum_{k=t}^{T-1} r_k$ <br>
#         $\quad\quad$ $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha ~ \underbrace{G ~ \nabla_{\boldsymbol{\theta}} \ln \pi_{\boldsymbol{\theta}}(\boldsymbol{a}_t|\boldsymbol{s}_t)}_{\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})}$ <br>
# <br>
# <b>RETURN</b> $\boldsymbol{\theta}$

# %% [markdown]
# ### Exercise 1: REINFORCE for discrete action spaces (Cartpole)

# %% [markdown]
# We will continue to work on the CartPole-v1 environment (c.f. https://gymnasium.farama.org/environments/classic_control/cart_pole/) which offers a continuous state space and discrete action space.
#
# Reminder:
# The Cart Pole task consists in maintaining a pole in a vertical position by moving a cart on which the pole is attached with a joint.
# No friction is considered.
# The task is supposed to be solved if the pole stays up-right (within 15 degrees) for 200 steps in average over 100 episodes while keeping the cart position within reasonable bounds.
# The state is given by $\{x,\frac{\partial x}{\partial t},\omega,\frac{\partial \omega}{\partial t}\}$ where $x$ is the position of the cart and $\omega$ is the angle between the pole and vertical position.
# There are only two possible actions: $a \in \{0, 1\}$ where $a = 0$ means "push the cart to the LEFT" and $a = 1$ means "push the cart to the RIGHT".

# %% [markdown]
# #### Policy Implementation
#
# We will implement a stochastic policy to control the cart using a simple one-layer neural network. Given the simplicity of the problem, a single layer will suffice. We will not incorporate a bias term in this layer.
#
# This neural network will output the probabilities of each possible action (in this case, there are only two actions: "push left" or "push right") given the input vector $s$ (the 4-dimensional state vector).

# %% [markdown]
# **Task 1.1**: Implement the `PolicyNetwork`  defined as follow.

# %% [markdown]
# The network takes an input tensor representing the state of the environment and outputs a tensor of action probabilities.
# The network has the following components:
#
# - `layer1`: This is a linear (fully connected) layer that takes `n_observations` as input and outputs `n_actions`. It does not include a bias term.
#
# - `forward` method: This method defines the forward pass of the network. It takes a state tensor as input and returns a tensor of action probabilities. It first applies the linear layer to the input state tensor to get the logits (the raw, unnormalized scores for each action), and then applies the softmax function to the logits to get the action probabilities. The softmax function ensures that the action probabilities are positive and sum to 1, so they can be interpreted as probabilities.
#
# This network is quite simple and may not perform well on complex tasks with large state or action spaces. However, it can be a good starting point for simple reinforcement learning tasks, and can be easily extended with more layers or different types of layers (such as convolutional layers for image inputs) to handle more complex tasks.

# %%
class PolicyNetwork(torch.nn.Module):
    """
    A neural network used as a policy for the REINFORCE algorithm.

    Attributes
    ----------
    layer1 : torch.nn.Linear
        A fully connected layer.

    Methods
    -------
    forward(state: torch.Tensor) -> torch.Tensor
        Define the forward pass of the PolicyNetwork.
    """

    def __init__(self, n_observations: int, n_actions: int):
        """
        Initialize a new instance of PolicyNetwork.

        Parameters
        ----------
        n_observations : int
            The size of the observation space.
        n_actions : int
            The size of the action space.
        """
        super(PolicyNetwork, self).__init__()

        ### BEGIN SOLUTION ###

        # self.layer1 = ...

        ### END SOLUTION ###


    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate the probability of each action for the given state.

        Parameters
        ----------
        state_tensor : torch.Tensor
            The input tensor (state).
            The shape of the tensor should be (N, dim),
            where N is the number of states vectors in the batch
            and dim is the dimension of state vectors.

        Returns
        -------
        torch.Tensor
            The output tensor (the probability of each action for the given state).
        """

        ### BEGIN SOLUTION ###

        # TODO...

        ### END SOLUTION ###

        return out


# %% [markdown]
# **Task 1.2**: Complete the `sample_discrete_action` function. This function is used to sample a discrete action based on a given state and a policy network. It first converts the state into a tensor and passes it through the policy network to get the parameters of the action probability distribution. Then, it creates a categorical distribution from these parameters and samples an action from this distribution. It also calculates the log probability of the sampled action according to the distribution. The function returns the sampled action and its log probability.

# %%
def sample_discrete_action(policy_nn: PolicyNetwork,
                           state: NDArray[np.float64]) -> Tuple[int, torch.Tensor]:
    """
    Sample a discrete action based on the given state and policy network.

    This function takes a state and a policy network, and returns a sampled action and its log probability.
    The action is sampled from a categorical distribution defined by the output of the policy network.

    Parameters
    ----------
    policy_nn : PolicyNetwork
        The policy network that defines the probability distribution of the actions.
    state : NDArray[np.float64]
        The state based on which an action needs to be sampled.

    Returns
    -------
    Tuple[int, torch.Tensor]
        The sampled action and its log probability.

    """

    ### BEGIN SOLUTION ###

    # Convert the state into a tensor, specify its data type as float32, and send it to the device (CPU or GPU).
    # The unsqueeze(0) function is used to add an extra dimension to the tensor to match the input shape required by the policy network.
    #state_tensor = ...

    # Pass the state tensor through the policy network to get the parameters of the action probability distribution.
    #actions_probability_distribution_params = ...

    # Create the categorical distribution used to sample an action from the parameters obtained from the policy network.
    # See https://pytorch.org/docs/stable/distributions.html#categorical
    #actions_probability_distribution = ...

    # Sample an action from the categorical distribution.
    #sampled_action_tensor = ...

    # Convert the tensor containing the sampled action into a Python integer.
    #sampled_action = ...

    # Calculate the log probability of the sampled action according to the categorical distribution.
    #sampled_action_log_probability = ...

    ### END SOLUTION ###
    
    # Return the sampled action and its log probability.
    return sampled_action, sampled_action_log_probability


# %% [markdown]
# **Task 1.3**: Test the `sample_discrete_action` function on a random state using an untrained policy network.

# %%
env = gym.make('CartPole-v1')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n.item()

### BEGIN SOLUTION ###

# TODO...

### END SOLUTION ###

print("state:", state)
print("theta:", theta)
print("sampled action:", action)
print("log probability of the sampled action:", action_log_probability)

env.close()


# %% [markdown]
# #### Implement the sample_one_episode function

# %% [markdown]
# Remember that in the REINFORCE algorithm, we need to generate a complete trajectory, denoted as $\tau = \{ \boldsymbol{s}_0, \boldsymbol{a}_0, r_1, \boldsymbol{s}_1, \boldsymbol{a}_1, \dots, r_T, \boldsymbol{s}_T \}$. This trajectory includes the states, actions, and rewards at each time step, as outlined in the algorithm at the beginning of Part 1.
#
# **Task 1.4**: Your task is to implement the `sample_one_episode` function. This function should play one episode using the given policy $\pi_\theta$ and return its rollouts. The function should adhere to a fixed horizon $T$, which represents the maximum number of steps in the episode.

# %%
def sample_one_episode(env: gym.Env,
                       policy_nn: PolicyNetwork,
                       max_episode_duration: int,
                       render: bool = False) -> Tuple[List[NDArray[np.float64]], List[int], List[float], List[torch.Tensor]]:
    """
    Execute one episode within the `env` environment utilizing the policy defined by the `policy_nn` parameter.

    Parameters
    ----------
    env : gym.Env
        The environment to play in.
    policy_nn : PolicyNetwork
        The policy neural network.
    max_episode_duration : int
        The maximum duration of the episode.
    render : bool, optional
        Whether to render the environment, by default False.

    Returns
    -------
    Tuple[List[NDArray[np.float64]], List[int], List[float], List[torch.Tensor]]
        The states, actions, rewards, and log probability of action for each time step in the episode.
    """
    state_t, info = env.reset()

    episode_states = []
    episode_actions = []
    episode_log_prob_actions = []
    episode_rewards = []
    episode_states.append(state_t)

    for t in range(max_episode_duration):

        if render:
            env.render_wrapper.render()

        ### BEGIN SOLUTION ###

        # Sample a discrete action and its log probability from the policy network based on the current state
        #action_t, log_prob_action_t = ...

        # Execute the sampled action in the environment, which returns the new state, reward, and whether the episode has terminated or been truncated
        #state_t, reward_t, terminated, truncated, info = ...

        # Check if the episode is done, either due to termination (reaching a terminal state) or truncation (reaching a maximum number of steps)
        done = terminated or truncated

        # Append the new state, action, action log probability and reward to their respective lists

        # TODO...

        ### END SOLUTION ###

        if done:
            break

    return episode_states, episode_actions, episode_rewards, episode_log_prob_actions


# %% [markdown]
# **Task 1.5:** Test this function on the untrained agent.

# %%
env = gym.make("CartPole-v1", render_mode='rgb_array')
RenderWrapper.register(env, force_gif=True)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n.item()

### BEGIN SOLUTION ###

#...
#episode_states, episode_actions, episode_rewards, episode_log_prob_actions = ...

### END SOLUTION ###

env.close()
env.render_wrapper.make_gif("lab2_reinforce_untained")

# %%
episode_states

# %%
episode_actions

# %%
episode_rewards


# %% [markdown]
# #### Implement a test function

# %% [markdown]
# **Task 1.6**: Implement the `avg_return_on_multiple_episodes` function that test the given policy $\pi_\theta$ on `num_episodes` episodes (for fixed horizon $T$) and returns the average reward on the `num_episodes` episodes.
#
# The function `avg_return_on_multiple_episodes` is designed to play multiple episodes of a given environment using a specified policy neural network and calculate the average return. It takes as input the environment to play in, the policy neural network to use, the number of episodes to play, the maximum duration of an episode, and an optional parameter to decide whether to render the environment or not. 
# In each episode, it uses the `sample_one_episode` function to play the episode and collect the rewards. The function then returns the average of these cumulated rewards.
#
# `avg_return_on_multiple_episodes` will be used for evaluating the performance of a policy over multiple episodes.

# %%
def avg_return_on_multiple_episodes(env: gym.Env,
                                    policy_nn: PolicyNetwork,
                                    num_test_episode: int,
                                    max_episode_duration: int,
                                    render: bool = False) -> float:
    """
    Play multiple episodes of the environment and calculate the average return.

    Parameters
    ----------
    env : gym.Env
        The environment to play in.
    policy_nn : PolicyNetwork
        The policy neural network.
    num_test_episode : int
        The number of episodes to play.
    max_episode_duration : int
        The maximum duration of an episode.
    render : bool, optional
        Whether to render the environment, by default False.

    Returns
    -------
    float
        The average return.
    """

    ### BEGIN SOLUTION ###

    # TODO...

    ### END SOLUTION ###

    return average_return


# %% [markdown]
# **Task 1.7:** Test this function on the untrained agent.

# %%
env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n.item()

### BEGIN SOLUTION ###

# TODO...

### END SOLUTION ###

print(average_return)

env.close()


# %% [markdown]
# #### Implement the train function

# %% [markdown]
# **Task 1.8**: Implement the `train_reinforce_discrete` function, used to train a policy network using the REINFORCE algorithm in the given environment. This function takes as input the environment, the number of training episodes, the number of tests to perform per episode, the maximum duration of an episode, and the learning rate for the optimizer.
#
# The function first initializes a policy network and an Adam optimizer. Then, for each training episode, it generates an episode using the current policy and calculates the return at each time step. It uses this return and the log probability of the action taken at that time step to compute the loss, which is the negative of the product of the return and the log probability. This loss is used to update the policy network parameters using gradient ascent.
#
# After each training episode, the function tests the current policy by playing a number of test episodes and calculating the average return. This average return is added to a list for monitoring purposes.
#
# The function returns the trained policy network and the list of average returns for each episode. This function encapsulates the main loop of the REINFORCE algorithm, including the policy update step. Please refer back to the algorithm outlined at the start of Part 1 for additional context, if necessary.

# %%
def train_reinforce_discrete(env: gym.Env,
                             num_train_episodes: int,
                             num_test_per_episode: int,
                             max_episode_duration: int,
                             learning_rate: float) -> Tuple[PolicyNetwork, List[float]]:
    """
    Train a policy using the REINFORCE algorithm.

    Parameters
    ----------
    env : gym.Env
        The environment to train in.
    num_train_episodes : int
        The number of training episodes.
    num_test_per_episode : int
        The number of tests to perform per episode.
    max_episode_duration : int
        The maximum length of an episode, by default EPISODE_DURATION.
    learning_rate : float
        The initial step size.

    Returns
    -------
    Tuple[PolicyNetwork, List[float]]
        The final trained policy and the average returns for each episode.
    """
    episode_avg_return_list = []

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n.item()

    policy_nn = PolicyNetwork(state_size, action_size).to(device)
    optimizer = torch.optim.Adam(policy_nn.parameters(), lr=learning_rate)

    for episode_index in tqdm(range(num_train_episodes)):

        ### BEGIN SOLUTION ###

        # TODO...

        ### END SOLUTION ###

        # Test the current policy
        test_avg_return = avg_return_on_multiple_episodes(env=env,
                                                          policy_nn=policy_nn,
                                                          num_test_episode=num_test_per_episode,
                                                          max_episode_duration=max_episode_duration,
                                                          render=False)

        # Monitoring
        episode_avg_return_list.append(test_avg_return)

    return policy_nn, episode_avg_return_list


# %% [markdown]
# #### Train the agent

# %%
env = gym.make('CartPole-v1')

num_trains = 3
reinforce_trains_result_list = [[], [], []]

for train_index in range(num_trains):

    # Train the agent
    reinforce_policy_nn, episode_reward_list = train_reinforce_discrete(env=env,
                                                                        num_train_episodes=250,
                                                                        num_test_per_episode=5,
                                                                        max_episode_duration=200,
                                                                        learning_rate=0.01)

    reinforce_trains_result_list[0].extend(range(len(episode_reward_list)))
    reinforce_trains_result_list[1].extend(episode_reward_list)
    reinforce_trains_result_list[2].extend([train_index for _ in episode_reward_list])

reinforce_trains_result_df = pd.DataFrame(np.array(reinforce_trains_result_list).T, columns=["num_episodes", "mean_final_episode_reward", "training_index"])
reinforce_trains_result_df["agent"] = "REINFORCE"

# Save the action-value estimation function of the last train

torch.save(reinforce_policy_nn, "reinforce_policy_network.pth")

env.close()

# %% [markdown]
# #### Plot results

# %%
g = sns.relplot(x="num_episodes", y="mean_final_episode_reward", kind="line", hue="agent", estimator=None, units="training_index", data=reinforce_trains_result_df, height=7, aspect=2, alpha=0.5)
plt.savefig("lab2_reinforce_cartpole_trains_result.png")

# %%
g = sns.relplot(x="num_episodes", y="mean_final_episode_reward", hue="agent", kind="line", data=reinforce_trains_result_df, height=7, aspect=2)
plt.savefig("lab2_reinforce_cartpole_trains_result_agg.png")

# %% [markdown]
# #### Test final policy

# %%
env = gym.make("CartPole-v1", render_mode='rgb_array')
RenderWrapper.register(env, force_gif=True)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n.item()

episode_states, episode_actions, episode_rewards, episode_log_prob_actions = sample_one_episode(env, reinforce_policy_nn, 200, render=True)

env.close()
env.render_wrapper.make_gif("lab2_reinforce_tained")


# %% [markdown]
# **Task 1.9**: decrease the learning rate value (e.g. 0.001), increase the number of episodes per training and retrain the agent. What do you observe ?

# %% [markdown]
# ### Exercise 2: REINFORCE with Baseline
#
# In the basic REINFORCE algorithm, the policy parameters are updated in proportion to the product of the gradient of the policy log-probability and the cumulative reward (return) from a state-action pair. However, this approach can lead to high variance in policy updates, making learning slower and less stable.
#
# The baseline is introduced to reduce this variance. It is a value subtracted from the cumulative reward when calculating the policy gradient. The key property of the baseline is that it does not affect the expected value of the gradient estimate, which means it doesn't bias the learning process but reduces the variance of the updates.
#
# The baseline can be thought of as a reference point or an "average" expectation of reward. By comparing the actual rewards to this baseline, we can determine whether the outcomes of certain actions are better or worse than this "average" performance.
#
# A common choice for the baseline is the value function of the current policy, $\hat{V}_{\boldsymbol{\omega}}$. By using the value function as a baseline, the algorithm adjusts the policy towards actions that perform better than the average.
#
# To incorporate the baseline into REINFORCE, you modify the update rule. Instead of using the total return $G$​ directly, you subtract the baseline value $\hat{V}_{\boldsymbol{\omega}}$ from $G$​ in the policy gradient estimate.
#
# By centering the rewards around a baseline, the variance of the policy gradient estimates is reduced. This leads to more stable and efficient learning, as the updates are less noisy and more focused on improving relative to the average performance.

# %% [markdown]
# #### REINFORCE with Baseline
#
# <b>REQUIRE</b> <br>
#  $\quad$ A differentiable policy $\pi_{\boldsymbol{\theta}}$ <br>
#  $\quad$ A differentiable baseline function $\hat{V}_{\boldsymbol{\omega}}(\boldsymbol{s})$ <br>
#  $\quad$ A learning rate $\alpha_1 \in \mathbb{R}^+$ for the policy <br>
#  $\quad$ A learning rate $\alpha_2 \in \mathbb{R}^+$ for the baseline <br>
# <b>INITIALIZATION</b> <br>
#  $\quad$ Initialize parameters $\boldsymbol{\theta} \in \mathbb{R}^d$ <br>
#  $\quad$ Initialize parameters $\boldsymbol{\omega} \in \mathbb{R}^d$ <br>
# <br>
# <b>FOR EACH</b> episode <br>
#  $\quad$ Generate full trace $\tau = \{ \boldsymbol{s}_0, \boldsymbol{a}_0, r_1, \boldsymbol{s}_1, \boldsymbol{a}_1, \dots, r_T, \boldsymbol{s}_T \}$ following $\pi_{\boldsymbol{\theta}}$ <br>
#  $\quad$ <b>FOR</b> $~ t=0,\dots,T-1$ <br>
#   $\quad\quad$ $G \leftarrow \sum_{k=t}^{T-1} r_k$ <br>
#   $\quad\quad$ $\delta_t \leftarrow G - \hat{V}_{\boldsymbol{\omega}}(\boldsymbol{s}_t)$ <br>
#   $\quad\quad$ $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha_1 ~ \delta_t ~ \nabla_{\boldsymbol{\theta}} \ln \pi_{\boldsymbol{\theta}}(\boldsymbol{a}_t|\boldsymbol{s}_t)$ <br>
#   $\quad\quad$ $\boldsymbol{\omega} \leftarrow \boldsymbol{\omega} + \alpha_2 ~ \delta_t \nabla_{\boldsymbol{\omega}}\hat{V}_{\boldsymbol{\omega}}(\boldsymbol{s}_t) $ <br>
# <br>
# <b>RETURN</b> $\boldsymbol{\theta}$

# %% [markdown]
# **Task 1.10**: Implement the `ValueNetwork` ($\hat{V}_{\boldsymbol{\omega}}$ in the algorithm) defined as follow.

# %% [markdown]
# `ValueNetwork` is a two-layer fully connected neural network. It takes an input tensor representing the state of the environment and outputs a tensor representing the estimated value of that state. The input tensor's shape should be (N, dim), where N is the number of state vectors in the batch and dim is the dimension of the state vectors.
#
# The network has the following components:
# - `layer1`: This is a linear (fully connected) layer that takes `n_observations` as input and outputs `nn_l1` neurons.
# - `layer2`: This is another linear layer that takes `nn_l1` neurons as input and outputs a single value.
# - `forward` method: This method defines the forward pass of the network. It takes a state tensor as input and returns a tensor representing the estimated value of the state. It first applies the ReLU activation function to the output of the first layer, and then applies the second linear layer to get the final output.
#
# This network is quite simple and may not perform well on complex tasks with large state spaces. However, it can be a good starting point for simple reinforcement learning tasks, and can be easily extended with more layers or different types of layers (such as convolutional layers for image inputs) to handle more complex tasks.

# %%
class ValueNetwork(torch.nn.Module):
    """
    A two-layer fully connected network that estimates the value of a state.

    Parameters
    ----------
    n_observations : int
        The number of observations in the state.
    nn_l1 : int, optional
        The number of neurons in the first layer, by default 16

    Attributes
    ----------
    layer1 : torch.nn.Linear
        The first fully connected layer.
    layer2 : torch.nn.Linear
        The second fully connected layer.
    """

    def __init__(self, n_observations: int, nn_l1: int = 16):
        super(ValueNetwork, self).__init__()

        ### BEGIN SOLUTION ###

        #self.layer1 = ...
        #self.layer2 = ...

        ### END SOLUTION ###


    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        state_tensor : torch.Tensor
            The input tensor representing the state.

        Returns
        -------
        torch.Tensor
            The output tensor representing the value of the state.

        """

        ### BEGIN SOLUTION ###

        # TODO...

        ### END SOLUTION ###

        return x


# %% [markdown]
# #### Implement the train function

# %% [markdown]
# **Task 1.11**: Implement the `train_reinforce_baseline_discrete` function, used to train a policy network and a value network using the REINFORCE with baseline algorithm in a given environment.
#
# The function first initializes a policy network and a value network, along with their respective Adam optimizers. Then, for each training episode, it generates an episode using the current policy and calculates the return at each time step. It uses this return, the log probability of the action taken at that time step, and the estimated value of the state to compute the policy and value losses. These losses are used to update the policy and value network parameters using gradient ascent. The value loss is typically defined as the squared difference between the estimated return and the actual return.

# %%
def train_reinforce_baseline_discrete(env: gym.Env,
                                      num_train_episodes: int,
                                      num_test_per_episode: int,
                                      max_episode_duration: int,
                                      policy_learning_rate: float,
                                      value_learning_rate: float) -> Tuple[PolicyNetwork, List[float]]:
    """
    Train a policy using the REINFORCE with baseline algorithm.

    Parameters
    ----------
    env : gym.Env
        The environment to train in.
    num_train_episodes : int
        The number of training episodes.
    num_test_per_episode : int
        The number of tests to perform per episode.
    max_episode_duration : int
        The maximum length of an episode.
    policy_learning_rate : float
        The policy learning rate.
    value_learning_rate : float
        The value learning rate.

    Returns
    -------
    Tuple[PolicyNetwork, List[float]]
        The final trained policy and the average returns for each episode.
    """
    episode_avg_return_list = []

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n.item()

    policy_nn = PolicyNetwork(state_size, action_size).to(device)
    policy_optimizer = torch.optim.Adam(policy_nn.parameters(), lr=policy_learning_rate)

    value_nn = ValueNetwork(state_size).to(device)
    value_optimizer = torch.optim.Adam(value_nn.parameters(), lr=value_learning_rate)

    for episode_index in tqdm(range(num_train_episodes)):

        # Generate an episode following the current policy
        episode_state_list, _, episode_reward_list, episode_log_prob_action_list = sample_one_episode(env=env,
                                                                                                      policy_nn=policy_nn,
                                                                                                      max_episode_duration=max_episode_duration)

        ### BEGIN SOLUTION ###

        # Initialize the value_loss to 0 at the beginning of each episode
        value_loss = 0

        # Iterate over the episode
        for t in range(len(episode_reward_list)):
            # Calculate the return at time t
            #future_return = ...

            # Convert the future_return to a PyTorch tensor
            #returns_tensor = ...

            # Convert the episode_log_prob_action_list[t] to a PyTorch tensor
            #log_prob_actions_tensor = ...

            # Convert the episode_state_list[t] to a PyTorch tensor
            #state_tensor = ...

            # Compute the baseline
            #delta = ...

            # Compute the policy loss
            #policy_loss = ...

            # Update the policy network
            # TODO...

            # Accumulate the value_loss
            # TODO...
        
        # Average the value_loss over the episode duration
        # TODO...

        # Update the value network
        # TODO...

        ### END SOLUTION ###

        # Test the current policy
        test_avg_return = avg_return_on_multiple_episodes(env=env,
                                                          policy_nn=policy_nn,
                                                          num_test_episode=num_test_per_episode,
                                                          max_episode_duration=max_episode_duration,
                                                          render=False)

        # Monitoring
        episode_avg_return_list.append(test_avg_return)

    return policy_nn, episode_avg_return_list


# %% [markdown]
# #### Train the agent

# %%
env = gym.make('CartPole-v1')

num_trains = 3
reinforce_baseline_trains_result_list = [[], [], []]

for train_index in range(num_trains):

    # Train the agent
    reinforce_baseline_policy_nn, episode_reward_list = train_reinforce_baseline_discrete(env=env,
                                                                                          num_train_episodes=250,
                                                                                          num_test_per_episode=5,
                                                                                          max_episode_duration=200,
                                                                                          policy_learning_rate=0.02,
                                                                                          value_learning_rate=0.02)

    reinforce_baseline_trains_result_list[0].extend(range(len(episode_reward_list)))
    reinforce_baseline_trains_result_list[1].extend(episode_reward_list)
    reinforce_baseline_trains_result_list[2].extend([train_index for _ in episode_reward_list])

reinforce_baseline_trains_result_df = pd.DataFrame(np.array(reinforce_baseline_trains_result_list).T, columns=["num_episodes", "mean_final_episode_reward", "training_index"])
reinforce_baseline_trains_result_df["agent"] = "REINFORCE baseline"

# Save the action-value estimation function of the last train

torch.save(reinforce_baseline_policy_nn, "reinforce_baseline_policy_network.pth")

env.close()

# %% [markdown]
# #### Plot results

# %%
g = sns.relplot(x="num_episodes", y="mean_final_episode_reward", kind="line", hue="agent", estimator=None, units="training_index", data=reinforce_baseline_trains_result_df, height=7, aspect=2, alpha=0.5)
plt.savefig("lab2_reinforce_cartpole_trains_result.png")

# %%
g = sns.relplot(x="num_episodes", y="mean_final_episode_reward", hue="agent", kind="line", data=reinforce_baseline_trains_result_df, height=7, aspect=2)
plt.savefig("lab2_reinforce_cartpole_trains_result_agg.png")

# %% [markdown]
# #### Test final policy

# %%
env = gym.make("CartPole-v1", render_mode='rgb_array')
RenderWrapper.register(env, force_gif=True)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n.item()

episode_states, episode_actions, episode_rewards, episode_log_prob_actions = sample_one_episode(env, reinforce_policy_nn, 200, render=True)

env.close()
env.render_wrapper.make_gif("lab2_reinforce_tained")

# %% [markdown]
# ### Bonus Exercise: Implementing REINFORCE for Continuous Action Spaces (Lunar Lander)
#
# The REINFORCE agent we've implemented so far is designed for environments with discrete action spaces. However, policy-based methods can effectively handle large, even continuous, action spaces. Instead of calculating learned probabilities for each possible action, we learn the statistics of the probability distribution. For instance, if the action set comprises real numbers, actions could be chosen from a normal (Gaussian) distribution.
#
# To create a policy parameterization, we can define the policy as the normal probability density over a real-valued scalar action. The mean and standard deviation of this distribution are determined by parametric function approximators (the `PolicyNetwork` neural network) that depend on the state.
#
# We can divide the policy’s parameter vector, $\boldsymbol{\theta} = [ \boldsymbol{\theta}_\mu, \boldsymbol{\theta}_\sigma ]^\top$, into two parts: one for approximating the mean and the other for approximating the standard deviation.
#
# **Task 1.12**: Modify the `PolicyNetwork`, `sample_discrete_action`, and `train_reinforce_discrete` functions to make REINFORCE compatible with the *LunarLander-v2* environment. Remember to set `continuous=True` in the `gym.make` function.

# %% [markdown]
# ## Part 2: Actor Critic

# %% [markdown]
# ### Actor Critic with bootstrapping
#
# <b>REQUIRE</b> <br>
#  $\quad$ A policy $\pi_{\boldsymbol{\theta}}$ and a value function $V_{\boldsymbol{\omega}}$ <br>
#  $\quad$ A learning rate $\alpha_1$ for the critic and $\alpha_2$ for the actor <br>
# <b>INITIALIZATION</b> <br>
#  $\quad$ $\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_n)$ <br>
#  $\quad$ $\boldsymbol{\omega} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_n)$ <br>
# <br>
# <b>FOR EACH</b> episode <br>
#  $\quad$ $\boldsymbol{s} \leftarrow \text{env.reset}()$ <br>
#  $\quad$ <b>DO</b> <br>
#   $\quad\quad$ $\boldsymbol{a} \sim \pi_{\boldsymbol{\theta}}(\cdot | \boldsymbol{s})$ <br>
#   $\quad\quad$ $r, \boldsymbol{s'} \leftarrow \text{env.step}(\boldsymbol{a})$ <br>
#   $\quad\quad$ $\boldsymbol{\omega} \leftarrow \boldsymbol{\omega} + \alpha_1 \left[ r + \gamma \hat{V}_{\boldsymbol{\omega}}(\boldsymbol{s'}) - \hat{V}_{\boldsymbol{\omega}}(\boldsymbol{s}) \right] \nabla_{\boldsymbol{\omega}} \hat{V}_{\boldsymbol{\omega}}(\boldsymbol{s})$ <br>
#   $\quad\quad$ $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha_2 \left[ \nabla_{\boldsymbol{\theta}} ~ \ln \pi_{\boldsymbol{\theta}}(\boldsymbol{a}|\boldsymbol{s}) \times \hat{V}_{\boldsymbol{\omega}}(\boldsymbol{s}) \right]$ <br>
#   $\quad\quad$ $\boldsymbol{s} \leftarrow \boldsymbol{s'}$ <br>
#  $\quad$ <b>UNTIL</b> $\boldsymbol{s}$ is final <br>
# <br>
# <b>RETURN</b> $\boldsymbol{\theta}$ <br>

# %%
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make('CartPole-v1')

# Neural networks
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layer(x)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layer(x)

actor = Actor()
critic = Critic()

# Hyper params
learning_rate = 0.001
optimizer_actor = optim.Adam(actor.parameters(), lr=learning_rate)
optimizer_critic = optim.Adam(critic.parameters(), lr=learning_rate)
gamma = 0.99

# Train loop
for episode in range(500):
    state, info = env.reset()
    episode_reward = 0
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state)
        action_probs = actor(state_tensor)
        action = torch.multinomial(action_probs, 1).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state_tensor = torch.FloatTensor(next_state)
        done = terminated or truncated
        episode_reward += reward

        # Update the Critic
        td_target = reward + gamma * critic(next_state_tensor) * (1 - int(done))
        td_error = td_target - critic(state_tensor)
        critic_loss = td_error.pow(2)
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

        # Update the Actor
        actor_loss = -torch.log(action_probs[action]) * td_error.detach()
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        state = next_state

    print(f'Episode {episode} : {episode_reward}')
