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

# %% [markdown] id="m-JpVXIi35eH"
# # INF639 Lab4: Inverse Reinforcement Learning
#
# <img src="https://raw.githubusercontent.com/jeremiedecock/polytechnique-inf581-2023/master/logo.jpg" style="float: left; width: 15%" />
#
# [INF639-2023](https://moodle.polytechnique.fr/course/view.php?id=17866) Lab session #4
#
# 2022-2023 Mohamed ALAMI

# %% [markdown]
# [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeremiedecock/polytechnique-inf639-2023-students/blob/master/lab4_irl.ipynb)
#
# [![My Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jeremiedecock/polytechnique-inf639-2023-students/master?filepath=lab4_irl.ipynb)
#
# [![NbViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/jeremiedecock/polytechnique-inf639-2023-students/blob/master/lab4_irl.ipynb)
#
# [![Local](https://img.shields.io/badge/Local-Save%20As...-blue)](https://github.com/jeremiedecock/polytechnique-inf639-2023-students/raw/master/lab4_irl.ipynb)

# %% [markdown] id="zT7OAPcv7SaX"
# ## Introduction
#
# The purpose of this lab is to introduce some classic algorithms of Imitation learning and Inverse Reinforcement Learning. We will see how they work, their caveats and benefits.
#
# You can either:
# - open, edit and execute the notebook in *Google Colab* following this link: https://colab.research.google.com/github/jeremiedecock/polytechnique-inf639-2023-students/blob/master/lab4_irl.ipynb ; this is the **recommended** choice as you have nothing to install on your computer
# - open, edit and execute the notebook in *MyBinder* (if for any reason the Google Colab solution doesn't work): https://mybinder.org/v2/gh/jeremiedecock/polytechnique-inf639-2023-students/master?filepath=lab4_irl.ipynb
# - download, edit and execute the notebook on your computer if Python3 and JypyterLab are already installed: https://github.com/jeremiedecock/polytechnique-inf639-2023-students/raw/master/lab4_irl.ipynb
#
# If you work with Google Colab or MyBinder, **remember to save or download your work regularly or you may lose it!**

# %% [markdown]
# ## Setup the Python environment

# %% [markdown]
# ### Install required libraries

# %% colab={"base_uri": "https://localhost:8080/"} id="MA9Rzi58CJpl" outputId="de779575-b2dd-48ea-a73f-42c302a66b1c"
# These installs are necessary for packages compatibility and visualization
# !sudo apt install swig
# !apt-get install -y xvfb x11-utils
# !pip install pyvirtualdisplay==0.2.* PyOpenGL==3.1.* PyOpenGL-accelerate==3.1.*
# !apt-get update && apt-get install ffmpeg freeglut3-dev xvfb
# !pip install stable-baselines3[extra] pyglet==1.5.27
# !pip install sb3_contrib
# !pip install box2d-py
# !pip install gym[box2d]==0.25

# %% [markdown]
# ### Import required packages

# %% colab={"base_uri": "https://localhost:8080/"} id="gWyX3YkJDXAk" outputId="bb09a4b7-0d1a-4af1-8294-913ec782e647"
import stable_baselines3
print(stable_baselines3.__version__)
import gym
print(gym.__version__)
from gym import spaces
import numpy as np
import time
from tqdm import trange, tqdm_notebook
import matplotlib.pyplot as plt
from IPython import display
import matplotlib
from matplotlib.pyplot import figure
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import pyvirtualdisplay
from typing import List, Tuple, Dict, Any, Callable, Optional
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader


# %% [markdown]
# ## Define utility functions
#
# These functions are necessary to produce videos of your agent at work.

# %%
def update_scene(num: int, frames: List, patch) -> Tuple:
    """
    Update the scene for animation.

    Parameters
    ----------
    num : int
        The frame number.
    frames : list
        The list of frames.
    patch : matplotlib object
        The image to b updated.

    Returns
    -------
    tuple
        A tuple containing the updated image.
    """
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames: List, repeat: bool = False, interval: int = 40) -> FuncAnimation:
    """
    Plot an animation.

    Parameters
    ----------
    frames : list
        The list of frames.
    repeat : bool, optional
        Whether to repeat the animation, by default False.
    interval : int, optional
        The interval between frames, by default 40.

    Returns
    -------
    FuncAnimation
        The animation object.
    """
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim


def generate_video(env: gym.Env, model: torch.nn.Module, n_timesteps: int = 300, initial_state=None) -> matplotlib.animation.ArtistAnimation:
    """
    Generate a video of the environment based on the model's actions.

    Parameters
    ----------
    env : gym.Env
        The environment to generate the video from.
    model : torch.nn.Module
        The model that predicts the actions.
    n_timesteps : int, optional
        The number of timesteps to run the simulation for, by default 300.
    initial_state : array_like, optional
        The initial state to start the simulation from, by default None.

    Returns
    -------
    matplotlib.animation.ArtistAnimation
        The generated video as a matplotlib animation.

    Raises
    ------
    Exception
        If the model fails to predict an action.
    """
    if initial_state is None:
        obs = env.reset()
    else:
        env = env.unwrapped  # to access the inner functionalities of the class
        env.state = initial_state
        obs = env.state

    figure(figsize=(8, 6), dpi=80)

    # use False con Xvfb
    _display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
    _ = _display.start()
    #model = model.float()

    frames = []

    for t in range(n_timesteps):
        #BEGIN ANSWER

        # TODO: ... (here you have to play for n_timesteps)

        #END ANSWER

        frame = env.render(mode='rgb_array')
        frames.append(frame)
        #env.render()
        time.sleep(.025)

    anim = plot_animation(frames)
    return anim


# %% [markdown] id="I1ADZUAr6mRB"
# ## Part 1: Behavioral Cloning
#
# *Behavioral Cloning* ([D. A. Pomerleau, *Efficient Training of Artificial Neural Networks for Autonomous Navigation*, Neural Computation, vol. 3, no. 1, pp. 88–97, 1991](https://cours.etsmtl.ca/sys843/REFS/ORG/pomerleau_alvinn.pdf)) represents the most fundamental approach to imitation learning. The principle is simple: an *expert* plays a game perfectly and the learning agent has just to act as a copy cat. For each state, it has to predict what the expert would have done.
# Usually the expert demonstrations are typically retrieved from recording human behaviour, but for the purpose of the exercise, we well consider fully trained RL models as expert and we will try to recover their optimal policy through behavioral cloning.
#
# The algorithm is recalled below.
#
# <b>Input</b>:<br>
# 	$\quad\quad$ Environment and expert model<br>
# <b>Algorithm parameter</b>:<br>
# 	$\quad\quad$ number of episodes $n$; number of epochs: epochs<br>
# <br>
# Train the expert model and get $\pi_{\text{exp}}$ <br>
# For episode in range($n$): <br>
# $\quad$ Initialise environment at state $s$ <br>
# $\quad$ While not done: <br>
# $\quad \quad$ action $a = \pi_{exp}(s)$ <br>
# $\quad \quad$  $s_{1}$, done = env($a$) <br>
# $\quad \quad$ store ($s,a$) in dictionary<br>
# $\quad \quad$ $s=s_1$ <br>
#
# Initialise model $\pi$ <br>
# For $k$ in range(epochs) <br>
# $\quad$ Train $\pi$ for each $(s,a)$ in dictionary such as $\pi(s)=a$
#
# You can find additional information about *Behavioral Cloning* in section 18.1 of *Algorithms for decision making* by M.J. Kochenderfer, T.A. Wheeler and K.H. Wray at MIT press (freely available online at https://algorithmsbook.com/files/dm.pdf).

# %% [markdown]
# ### Import required packages

# %% [markdown] id="RSYfV4427C7W"
# #### Cartpole
#
# We will use The Cartpole environment as a first example. More details on this environment are available [here](https://gymnasium.farama.org/environments/classic_control/cart_pole/).

# %% colab={"base_uri": "https://localhost:8080/"} id="pmPKOnOi687p" outputId="aacdb084-46f7-47c3-cbb7-f91891880a44"
env = gym.make("CartPole-v1", render_mode="rgb_array")

# %% [markdown] id="i9IY4lfnIILz"
# The aim of this lab is not to train the expert so we can either consider models that are already trained on Cartpole or use Stable_baselines, a framework that allows to train several heavily tested famous algorithms in one line. More details [here](https://stable-baselines.readthedocs.io/en/master/).
#
# You can either train the expert model or load a pretrained model.
#
# Uncomment the following lines if you want to train the expert model (should take 10 min max):

# %% id="57vTcyKbDIBk"
## We first initialise the model by saying that we will use the PPO algorithm and use a classic
## MLP parameterisation
# cartpole_expert = PPO(MlpPolicy, env, verbose=1)

## Then the model can easily be trained in just one line
# cartpole_expert.learn(total_timesteps=250000)

## You can save or load your model easily
# cartpole_expert.save("lab4_cartpole.zip")

# %% [markdown] id="HS68MFcJlMyV"
# Otherwise, uncomment the following cell to load a pretrained expert model:

# %% colab={"base_uri": "https://localhost:8080/"} id="3AHcbEndhG6T" outputId="b9874121-f468-4691-aeac-26704d053c36"
# #!wget https://github.com/jeremiedecock/polytechnique-inf639-2023-students/raw/master/lab4_cartpole.zip

# %% colab={"base_uri": "https://localhost:8080/"} id="3ybJJn43-uLf" outputId="67e42b3c-c593-49ea-87fb-190e35e9ab7b"
cartpole_expert = PPO.load("lab4_cartpole.zip")

# %% colab={"base_uri": "https://localhost:8080/", "height": 593} id="jh08UFKLRAoJ" outputId="905c6e70-5834-4291-b472-33c60f58cd39"
env = gym.make('CartPole-v1', new_step_api=True)
anim = generate_video(env, cartpole_expert)
HTML(anim.to_html5_video())


# %% [markdown] id="jvKTBMqcLHuV"
# This function is used to create expert trajectories that will be used in our Behavior Cloning model

# %%
def generate_expert_traj(model: torch.nn.Module, env: gym.Env, n_episodes: int = 100) -> Dict[str, np.array]:
    """
    Generate expert trajectories using the provided model and environment.

    Parameters
    ----------
    model : torch.nn.Module
        The model that predicts the actions.
    env : gym.Env
        The environment to generate the trajectories from.
    n_episodes : int, optional
        The number of episodes to run the simulation for, by default 100.

    Returns
    -------
    dict
        A dictionary containing the actions, observations, rewards, episode returns, and episode starts.
    """
    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []  #Index of new episodes initial states

    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0

    for ep_idx in tqdm_notebook(range(n_episodes), desc='Epoch', leave=True):

        #BEGIN ANSWER

        # TODO: ... (play the game using your expert model and store observations, actions and rewards in lists)

        #END ANSWER

    if isinstance(env.observation_space, spaces.Box):
        observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))

    if isinstance(env.action_space, spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        actions = np.array(actions).reshape((-1, 1))

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)

    env.close()

    return numpy_dict


# %% colab={"base_uri": "https://localhost:8080/", "height": 192, "referenced_widgets": ["1054ad01c4c84c79a50121c34f92ca0f", "4e79a431cba74808a6ba52058c6f5be7", "f5405cc2efd34208b7e42ecc75e61423", "dbb20487c06447c89e468e83b3193321", "7da9a564c93c49cb94a16a74a75e40af", "ba46fe4aa6914a95bc1e1bf73c0d6479", "98229c1991444139956b194d6a332cd5", "d05afa2beb7e40318c19171c8582cb3d", "349c241c901b4ff49e93f5a8d64e333c", "57ab2c6c74954f419fcd5e9a04ce190b", "528fac4fb01943e6a8eab867d6394d04"]} id="rPyUWIuzKFjf" outputId="ef47d48b-0129-4a5f-8f16-6fec82c90f92"
# Generate expert trajectories
cartpole_demos = generate_expert_traj(cartpole_expert, env)


# %% [markdown]
# The Behavioral Cloning model is defined in the following cell.

# %%
class Net(nn.Module):
    """
    A feed-forward neural network.

    Attributes
    ----------
    fc1 : torch.nn.Linear
        The first fully connected layer.
    fc2 : torch.nn.Linear
        The second fully connected layer.
    fc3 : torch.nn.Linear
        The third fully connected layer.

    Methods
    -------
    forward(x)
        Passes the input through the network.
    """

    def __init__(self):
        """
        Initializes the layers of the network.
        """
        super(Net, self).__init__()

        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Passes the input through the network.

        Parameters
        ----------
        x : torch.Tensor
            The input to the network.

        Returns
        -------
        torch.Tensor
            The output of the network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


# %% [markdown]
# This function transforms the dataset of expert state-actions pairs into a Dataset class. That makes it easier to train in batches

# %% id="d0fwywqAYbSP"
class NumpyDataset(data.Dataset):
    def __init__(self, expert_trajectories_dict: Dict[str, np.array], transform: Optional[Callable] = None):
        """
        Initialize the dataset.

        Parameters
        ----------
        array : Dict[str, torch.Tensor]
            The input data as a dictionary.
        transform : Optional[Callable], optional
            A function/transform that takes in an torch.Tensor and returns a transformed version.
        """
        super().__init__()
        self.obs = torch.FloatTensor(array["obs"])
        self.actions = torch.FloatTensor(expert_trajectories_dict["actions"])
        self.transform = transform

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.obs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the data at the given index.

        Parameters
        ----------
        index : int
            The index to get the data from.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The observation and action at the given index.
        """
        obs = self.obs[index]
        action = self.actions[index]
        if self.transform:
            obs = self.transform(obs)
            action = self.transform(action)
        return obs, action


# %% id="Iyrf5aTwKXP2"
loader_args = dict(batch_size=128, shuffle=True)
train_dset = NumpyDataset(cartpole_demos)
train_loader = data.DataLoader(train_dset, **loader_args)
device = "cpu"


# %% id="HXQX9pNWd3dD"
def plot_train_curves(epochs: int, train_losses: List[float], test_losses: Optional[List[float]] = None, title: str = '') -> None:
    """
    Plot training and testing losses over epochs.

    Parameters
    ----------
    epochs : int
        The number of epochs.
    train_losses : List[float]
        The list of training losses.
    test_losses : Optional[List[float]], optional
        The list of testing losses, by default None.
    title : str, optional
        The title of the plot, by default ''.
    """
    x = np.linspace(0, epochs, len(train_losses))
    plt.figure()
    plt.plot(x, train_losses, label='train_loss')
    if test_losses:
        plt.plot(x, test_losses, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()


def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> float:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        The DataLoader for the training data.
    criterion : nn.Module
        The loss function.
    optimizer : optim.Optimizer
        The optimizer.

    Returns
    -------
    float
        The average loss for this epoch.
    """
    # the criterion parameter is a placeholder for the loss function you might choose afterwards
    model.train()
    total_loss = 0

    for obs, action in train_loader:
        #BEGIN ANSWERS

        # TODO: ... (train the model and return the average loss)

        #END ANSWER

        return avg_loss.item()


def train_epochs(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, test_loader: Optional[DataLoader] = None, train_args: Optional[Dict[str, int]] = None, plot: bool = True) -> Tuple[List[float], List[float]]:
    """
    Train the model for multiple epochs.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        The DataLoader for the training data.
    criterion : nn.Module
        The loss function.
    test_loader : Optional[DataLoader], optional
        The DataLoader for the testing data, by default None.
    train_args : Optional[Dict[str, int]], optional
        The training arguments (epochs and learning rate), by default None.
    plot : bool, optional
        Whether to plot the training curve, by default True.

    Returns
    -------
    Tuple[List[float], List[float]]
        The training and testing losses for each epoch.
    """
    # training parameters
    epochs, lr = train_args['epochs'], train_args['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    for epoch in tqdm_notebook(range(epochs), desc='Epoch', leave=False):
        model.train()
        train_loss = train(model, train_loader, criterion, optimizer)
        train_losses.append(train_loss)

    if plot:
        plot_train_curves(epochs, train_losses, test_losses, title='Training Curve')
    return train_losses, test_losses


# %% colab={"base_uri": "https://localhost:8080/", "height": 526, "referenced_widgets": ["ca1b982ca67d4c1484277ab4d9364566", "f6b950af011340eb8bdd61ada2905776", "e19989312c394203836770ac7add4321", "57b6ea3a757d44eb8e036fb102ec3da0", "ff66d84c286c4d02872811ce0d579e21", "3a4f325b708d4c04a30bda1cdb5456f3", "72cff1af5b4d4bc892ee0312d45b4880", "d8720b7e0718402d9860b89b37706afe", "74889e7f20ae42b2b3846a566aa37580", "9073093d85a24bc3bbb5f6b0d0923d23", "905a957cf9344b388338725bc0959b2d"]} id="WgTr49L5fTDq" outputId="b81ec262-46b7-4684-fc54-40832bc76f61"
criterion = nn.BCELoss()

cartpole_BC_model = Net()
train_args = dict(epochs=3000, lr=0.0001)
train_losses, test_losses = train_epochs(cartpole_BC_model, train_loader, criterion, train_args=train_args)

# %% colab={"base_uri": "https://localhost:8080/", "height": 593} id="5fzgvaINAQT8" outputId="a79836c0-6f95-4aae-b2c4-8ef6897032c1"
anim = generate_video(env, cartpole_BC_model)
HTML(anim.to_html5_video())

# %% [markdown] id="faT8kikOQm1-"
# Even with a very quick training, we see that the agent is pretty good. But is it that good ?

# %% [markdown] id="RGzyAwx2GDZK"
# Now let's try with a different initial state

# %% colab={"base_uri": "https://localhost:8080/", "height": 628} id="tBjk1FSCGTze" outputId="ade34d83-e512-4f8d-f682-862a1f2302f9"
obs = env.reset()
state_init = np.array([1.0, obs[1], obs[2], obs[3]]) #we slightly move the initial position of the cart

anim = generate_video(env, cartpole_BC_model, initial_state=state_init)
HTML(anim.to_html5_video())

# %% [markdown] id="vX22iztXpYDW"
# When changing the initial state, we put the agent in a situation the expert has never seen; therefore the agent is lost and does not know what is the good action to perform.
#
# Actually it is not even necessary to change the initial state. The agent is never absolutely perfect, so with a long enough horizon, it will make some mistakes that stack up and drive it into states that the expert has never seen (as it only explores best states).  

# %% [markdown] id="0wAo43EsWNpA"
# ### Lunar Lander

# %% [markdown] id="b4-tmMeNl_wa"
# Let's test with the Lunar Lander environment. The documentation of this environment is available here: https://gymnasium.farama.org/environments/box2d/lunar_lander/.

# %% id="TqHmbeD3l1c3"
env = gym.make('LunarLander-v2')

# %% [markdown] id="g-z-kkvcqKur"
# You can either train the expert model or load a pretrained model.
#
# Uncomment the following lines if you want to train the expert model (should take 10 min max):

# %% colab={"base_uri": "https://localhost:8080/"} id="j1yTDDolzPQf" outputId="078b0565-9195-4af9-dcda-46d77702da13"
# model = PPO(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=250000)

# model.save("lab4_lander.zip")

# %% [markdown] id="OFlgE1JPlbJp"
# Otherwise, uncomment the following cell to load a pretrained expert model:

# %% colab={"base_uri": "https://localhost:8080/"} id="-wXAnYmGjZtF" outputId="89b00a47-0000-4de5-a309-17897fd118d7"
# #!wget https://github.com/jeremiedecock/polytechnique-inf639-2023-students/raw/master/lab4_lander.zip

# %% id="L3tKZP4L9YcI"
expert_lander = PPO.load("lab4_lander.zip")

# %% colab={"base_uri": "https://localhost:8080/", "height": 593} id="kMIJl_sQG9Zy" outputId="f6938470-600c-4966-b325-6fbf0b436d39"
env = gym.make('LunarLander-v2', new_step_api=True)
anim = generate_video(env, expert_lander, 400)
HTML(anim.to_html5_video())

# %% colab={"base_uri": "https://localhost:8080/", "height": 192, "referenced_widgets": ["2093b01cdada4f3680d436b13cc74332", "53302ba9de844bf6bc1ec2929c2c5cf8", "7b81b49e768a4951a981c696b2476dc0", "39012e0c7fdb4e358fc84e4b43d713c1", "42db98b7c69f4b12a8674c7af08fd7ad", "ae10596a21fc47508bead864a72870e5", "1490e8ff22444e178922e97ca9699445", "f830196e84454efd8c75987430e04883", "e847984d541b4afb9b9f6b36af76b46c", "eeb4aea7cb1f4c54810ee6e0ef374d11", "94ff44287d664ed78cd431197988c593"]} id="WVj0heG7_RNY" outputId="80aa62f9-af6f-4110-c28f-c8149e2d0596"
Lander_demos = generate_expert_traj(expert_lander, env)


# %% [markdown]
# The Behavioral Cloning model is defined in the following cell.

# %% id="OSCGmn5cUqKE"
class Lander_Net(nn.Module):
    """
    A feed-forward neural network for the Lunar Lander task.

    Attributes
    ----------
    fc1 : torch.nn.Linear
        The first fully connected layer.
    fc2 : torch.nn.Linear
        The second fully connected layer.
    fc3 : torch.nn.Linear
        The third fully connected layer.
    fc4 : torch.nn.Linear
        The fourth fully connected layer.
    fc5 : torch.nn.Linear
        The fifth fully connected layer.

    Methods
    -------
    forward(x)
        Passes the input through the network.
    """

    def __init__(self):
        """
        Initializes the layers of the network.
        """
        super(Lander_Net, self).__init__()

        self.fc1 = nn.Linear(8, 24)
        self.fc2 = nn.Linear(24, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the network.

        Parameters
        ----------
        x : torch.Tensor
            The input to the network.

        Returns
        -------
        torch.Tensor
            The output of the network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc5(x))

        return x


# %% colab={"base_uri": "https://localhost:8080/", "height": 561, "referenced_widgets": ["53e67d8255fc4d2bacb9440682fcb95e", "2d1e9f95fa60400f9f152952bc99b9a7", "5674310cc7244bdaba5d80eeb07715eb", "3e029a1d9234449cb9cca1703f7dc983", "bf8f09b963b14cfbb69d331172c18435", "56edc559b916456888fad60dfc50be31", "250adce852964863bff61bd11006ccd9", "60a98de39b444356852e5b48cd8bb516", "efaa63c675a9499182dc5dd680794954", "3f3164a9581e4379a1a1d68b29694cf4", "9e3270c647e54f5caeff103b606be20c"]} id="Hlxrzgg3cEJu" outputId="86f789e5-d285-4cf9-fd3d-3902c1c06682"
loader_args = dict(batch_size=128, shuffle=True)
train_dset = NumpyDataset(Lander_demos)
train_loader = data.DataLoader(train_dset, **loader_args)

#BEGIN ANSWER
criterion = ... # TODO: choose the right loss function
#END ANSWER

Lander_BC_model = Lander_Net()
train_args = dict(epochs=2000, lr=0.001)
train_losses, test_losses = train_epochs(Lander_BC_model, train_loader, criterion, train_args=train_args)

# %% colab={"base_uri": "https://localhost:8080/", "height": 628} id="_uijZQrMO1YU" outputId="4d811278-547b-490f-ccbe-9abd5e8fa797"
env = gym.make('LunarLander-v2', new_step_api=True)
anim = generate_video(env, Lander_BC_model, 500)
HTML(anim.to_html5_video())

# %% [markdown] id="nTNcDz9ciyCe"
# ### Lunar Lander stochastic

# %% [markdown] id="IvcpKbzLt-_3"
# Now let's try something else. We will add wind in the environment to make the dynamics of the environement stochastic.

# %% colab={"base_uri": "https://localhost:8080/"} id="HsdkpE4ni2HR" outputId="b3af37f8-0a99-479b-deb8-e873ca765e87"
env = gym.make('LunarLander-v2', enable_wind=True)

# %% [markdown] id="r5j8se6GmcGV"
# Again, you can either train the expert model or load a pretrained model.
#
# Uncomment the following lines if you want to train the expert model (should take 10 min max):

# %% id="dyQU07abIv5P"
# model = PPO(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=250000)
# model.save("lab4_windy_lander.zip")

# %% [markdown] id="N4NpPeY5m5xr"
# Or, uncomment the following cell to load a pretrained expert model:

# %% colab={"base_uri": "https://localhost:8080/"} id="JLGWe9TDm_6w" outputId="e91083f2-f84a-4333-f1d2-5bfae3d4c141"
# #!wget https://github.com/jeremiedecock/polytechnique-inf639-2023-students/raw/master/lab4_windy_lander.zip

# %% colab={"base_uri": "https://localhost:8080/"} id="jl1uLm1HI5Bt" outputId="380af4c6-2b0a-483b-be66-c17b117e39f8"
expert_windy_lander = PPO.load("lab4_windy_lander.zip")

# %% colab={"base_uri": "https://localhost:8080/", "height": 593} id="qE5YPxKyJDIy" outputId="c6d3e726-1ff4-4b4c-b342-c9d975d1d2a6"
env = gym.make('LunarLander-v2', enable_wind=True, new_step_api=True)
anim = generate_video(env, expert_windy_lander, 400)
HTML(anim.to_html5_video())

# %% colab={"base_uri": "https://localhost:8080/", "height": 192, "referenced_widgets": ["44208df2ffef4663a0930958d9530e0c", "3b8f003f35604981bb2e9b2288806760", "caf0aa0ed7b4456fb7cd607db557b0f5", "aeda022496c64af38ad1619847d44d27", "5bfd4013c34844dfa6e4eed9e4f3f106", "c19aa4ded37a4fdb90b03cc5329375f7", "e88caddb95e4483f80db9dedf4348335", "c1936849814843609acc1a7bce3f96c5", "748ed0d41ff64cf79a3af4ae7ecb3a01", "dd64f503d4d14e4595fc5b215ed3084b", "ecda092f699b416ba14cc45916178947"]} id="6HVtCbTXJlOT" outputId="61e5b17d-418f-4028-916b-b64825d52b66"
Lander_demos = generate_expert_traj(expert_windy_lander, env)

# %% colab={"base_uri": "https://localhost:8080/", "height": 561, "referenced_widgets": ["851b2f16215044209ded45c2aebe691b", "6576dc45fb964ee194b4ed4fc222da3c", "a85908f708924106bb3a7c86ab570220", "db80804dfc8745d1b6e0fad4c358140c", "84495ed2638045608d6554c39be03de6", "265496135a5442cdaba1930172e9ca1f", "7ab0b4d5295b431d989246d7883fb407", "185c88e974f44b89b40a455c88a68cae", "38c1aa844ebf422189482474439226d4", "e6fa5698c98e4fe29f71e55c5731972b", "08c5d6ba69784389948315eb5330a8ef"]} id="ZDo6AuOUKNkx" outputId="18a49158-9054-48d8-cc6d-ab006beffd8c"
device = "cpu"
loader_args = dict(batch_size=128, shuffle=True)
train_dset = NumpyDataset(Lander_demos)
train_loader = data.DataLoader(train_dset, **loader_args)

#BEGIN ANSWER
criterion = ... # TODO: choose the right loss function
#END ANSWER

Windy_Lander_BC_model = Lander_Net()
train_args = dict(epochs=2000, lr=0.001)
train_losses, test_losses = train_epochs(Windy_Lander_BC_model, train_loader, criterion, train_args=train_args)

# %% colab={"base_uri": "https://localhost:8080/", "height": 628} id="eUepPnixK5Oe" outputId="cf553bd7-a88f-46b4-c73e-fe231ae65e37"
anim = generate_video(env, Windy_Lander_BC_model, 500)
HTML(anim.to_html5_video())


# %% [markdown] id="qV-QMjpbwFX8"
# The agent fails because stochastic environments increases "mistakes" as well as the chances to end in non optimal states that the expert never encountered

# %% [markdown] id="2PfG6unTVhA8"
# ## Part 2: DAgger

# %% [markdown] id="pBEbe3hXwRHe"
# To mitigate that problem, a classic solution is to use DAgger algorithm ([S. Ross, G. J. Gordon, and J. A. Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, in International Conference on Artificial Intelligence and Statistics (AISTATS), vol. 15, 2011](http://proceedings.mlr.press/v15/ross11a/ross11a.pdf)). The idea is simple. We just have to create a feedback loop between the agent and the expert. After each roll or episode of the agent, its trajectory (the list of states) it encountered are given back to the expert, who attribute to them the right action to perform. The nex state-action pairs are then added to the original dataset, and a new training is done on the aggregated dataset

# %% [markdown] id="lR9yq7M7w3tS"
# Below is the algorithm's pseudocode: <br>
# <b>Input</b>:<br>
# environment, expert model, agent model, number of iterations, number of trajectories to sample from the agent at each iteration <br>
#
# <b>Algorithm</b>: <br>
#
# Initialise dataset <br>
# <b>FOR</b> $i$ in range(number of iterations): <br>
# $\quad$ traj $\xleftarrow{}$ generate model trajectories <br>
# $\quad$ get states from traj <br>
# $\quad$ get expert actions for states <br>
# $\quad$ add states and actions to dataset <br>
# $\quad$ train agent on dataset
#
#
# You can find additional information about *DAgger* in section 18.2 of *Algorithms for decision making* by M.J. Kochenderfer, T.A. Wheeler and K.H. Wray at MIT press (freely available online at https://algorithmsbook.com/files/dm.pdf).

# %%
def get_expert_actions(model: nn.Module, states: np.ndarray) -> np.ndarray:
    """
    Extract expert actions from given states.

    Parameters
    ----------
    model : nn.Module
        The model to predict actions.
    states : np.ndarray
        The states to predict actions for.

    Returns
    -------
    np.ndarray
        The predicted actions.
    """
    # BEGIN ANSWER
    actions = ...  # TODO
    # END ANSWER
    return actions

def DAgger(env: gym.Env, expert: nn.Module, model: nn.Module, n_iterations: int = 10, n_traj: int = 10, iter_plot: int = 1) -> None:
    """
    Implementation of the DAgger algorithm.

    Parameters
    ----------
    env : gym.Env
        The environment to perform the algorithm in.
    expert : nn.Module
        The expert model.
    model : nn.Module
        The model to train.
    n_iterations : int, optional
        The number of iterations to perform, by default 10.
    n_traj : int, optional
        The number of trajectories to generate, by default 10.
    iter_plot : int, optional
        The number of iterations between plots, by default 1.
    """
    dataset = {'obs': None, 'actions': None}
    losses = []
    # BEGIN ANSWER
    criterion = ...   # TODO
    # END ANSWER

    for i in tqdm_notebook(range(n_iterations)):
        # BEGIN ANSWER

        # TODO...

        train_losses, test_losses = ... # TODO

        # END ANSWER

        if losses == []:
            losses = train_losses
        else:
            losses = np.concatenate((losses, np.array(train_losses)))
        if n_iterations % iter_plot == 0:
            plt.plot(losses)
            plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["b7b9a271f4a0493194b3b6a8d1f168bd", "c472125c8737402a958f6249da5d8f51", "d67e41fb3ec445878c88eb33d2a9c28c", "4be506061a2e4db397c623c8ac0996a7", "819dbbe04859472e9326f795c46c8ef8", "fb12ddc2ca3e44b8b031f6f62b2b78f8", "c953016e0c004b7e8f242670a416f58e", "0811a64005824ed8ae5fcb2c69091a53", "5e19e0990d3d47a9bc8a69b422c76399", "5d5a2cf7c12e44e4b9135e9fcb7626ee", "af848737760f46b5a098c7e0d7da9b78", "2a8439ea8057418cb3f5094bc6d0d97b", "311d0e29e03c421bb331c1ccd1b0f3ec", "7957979366e141f6b788277c9985eb51", "6101f50f3f544ee4be705aaa294c7ced", "69d357e582fa4cd4a52f81f91514b26a", "27736d9decf549829a5c79bc076cd74f", "ed854de408434fa8b6a50411c214b7a3", "ae850f9e919449d697d15c4ba53ea058", "5c9f61d75a524a279224c0ab7f072c97", "040b41b52bbf42dc817e732d41714acb", "51f299190b2a4bdb9a901c071dae2bca", "ef334779e8aa4ba2827fa6e4aec25875", "e4dc4eec40aa473a93070cf7af640316", "6eeab0064cb1408a8aaeb60676b1dac3", "268f75ecaadd4792a0efbb6280b2bcfd", "e3cc05c30d914df395b018632a0786f6", "d8af62c067ed4eddbf0491b675e9b6f2", "b6e327113f074e6caaf2ae5b50318876", "9f591e689da94efca5cc64b91d1dcc97", "f8b7a21620cf415fb974798910c4d1d6", "19a8069c0a8f4e02a1501949ccd1b2c3", "599f03c34ab04fb9bc9304fcba8191dd", "0b9600ba7f0d4c4c834bf6d24391e8c8", "bc133ad5b98e465c9a91cbd31da4967c", "8af35fa70ba642858b2769a3001542e4", "ce188ea5cf044cc98ebc4715ab20e734", "87df756954624307be74f538ccfcbf80", "6b998add44c448f1bda42c4520bd68a2", "2ff194ec9dd244cca2c5da93187d9095", "87b105bc87d941879bf175f0c7ff883c", "3d0605fa143c49649215447ff48e6856", "9b0bcede51a1452ca62f1fa290d5f81b", "e3f443bd8f184f36887a16ddf435254b", "fa167e9e8c7a4d15a117f687dae0886d", "50cc96132fa34aefa380e842566d3d94", "2785ee4051584dcb96f709362ff65923", "96845b990f9d4cc6a163f0ea727dcb71", "64ec3e4909e74071ac783c41e20ca80e", "f7a0736fa8094b90bf9207ed6d8e872c", "4d855c4a2f7c4e2d9927f98db383280b", "e214e819ca224e4aa62570927cae992e", "9df960d44f494207a0e576a3ce0e3d5b", "8dea614b1e1a463d8b89e5b6b8a6efe6", "299b5b0284ec41459746be7ce4a7b6e0", "a96956053e534b58a3be374693bb4afd", "efd6584c2be8417290b5627dfd0316eb", "3434c222dd8d46e88a1de8166018c910", "3fc6946c248b45eea95d42d68f8a48e4", "ec563f46788443fb8c7bb958c0765210", "e31a1b3493944de493bb63cd9b28b06f", "57d2091165064f30986e686bc2973806", "65d20c4acae54755ab4367e9fcda0127", "6aa0d258209a432790bb7bcbefc76437", "f0aa50e02599432282c44389d2f4df86", "52d54f91a50049ef885c53a568bce53f", "b9b468e056d744dcaa1c699fe58a6097", "094af53b830044cdbf51c0b6b7f41a0c", "362fe9e6a51f4cbb88dbdea34d0fef64", "e66c8e1e5e074fc096c2f79e93f3bd60", "3cb615c3f2de42798136f012825535e6", "f95949c3bee14ce79806bb1ad0e9c24a", "944212cd5ca14fa38d44bc3954daeaaf", "0611bd135e0e4d92af6352e4a5ac1175", "d85d8db3aece4f7dbb250d4a53c15a4b", "88b10efb9df0408681e5fec19746e4ce", "878fe53efdae4f6591145c88c705b0f4", "b09b05780ff9459ca8ddc1138d39d841", "c2b7806cf9be441ab8f54bba2227edef", "c1ed7124c5fa47ccb5d47213551420d7", "a8d2afe7d90f417d90c68f941a5fbab0", "e851d0f1165740ffae2369307128f75d", "6751691ab8954729a8c2f033896cd6f7", "121157640e934d39874c7fadf4b87836", "395c18b6332f4942964e68c484eb89b8", "84f7ef8e73614e888f5e65fc319fcc21", "bb7b2ccc3443460b8eb1fab9c80cc166", "320077e628c4403bb2565e7288aa55a1", "33abec53f6f34091b297271209f5aad6", "0b9484248188443abe0c390611d82dba", "c84a108098704d21a03f0b9e55d6abe6", "247e59c4132342f28b56068dd76075ba", "72e15d9ada5b4a718680e7018b42ebce", "92e2f12e00d74277b9d28baa20adf68d", "116eeca977004bdd9704ba615d6cee7c", "3632e5307cfc4c2c9f84d1f5ee561425", "4e779ed1ef1a4a3f92e6342c58a93a0f", "930ed21ae5754b98af49ff8a16c2b647", "f001f2ef61a549d686330e6ed684570c", "e0b836b819e54066aa8c93c70a0888ae", "7ffca5b3003e47a3ad87c006071ac42d", "fcddfb82392e424680078de467fb26b7", "02fdcddc75e44326b843dbabf899489f", "04aa1b08eace411895ffcc24e803cb27", "8d9cce72ac1b46a19b37c2992e8cdfc9", "608e826d8c5f462aa7dbc6219b68ade5", "25cfc302c97b4a0fa14ebcd2737391fe", "540fbb55cd9c46f48ebcbb94930aafca", "28f4e7956b6a4852ae05d62d5fb870b1", "de1434fdc9e0456ca3c38563fe321c5f", "1ab679e4a23e4952bc556ec4bb6a62e3", "e72ccf9964a34d9a9ee11389c88069ed", "b72bde1393cd42ed97e9fcaa55081761", "5265d3ff41a449b3a5e5f2670097b247", "809aa8aa28544000be8901ffa98f8965", "1278252501d341f986bf3515e5902f6d", "2912eff27b6440f9b3a7d44c3b289d70", "4d0c82b9fd1c48b4901ce671704fb1c9", "7d6d2ece89084fd4a3cae21c66d484de", "9318ca308f454d82ab987a3240a2cffd", "cd69752b484543e98d818925079a43c5", "9d5a190dce304352a59c18381a49d35c", "76d3457ccfcd4fa38db8388a52a77921", "943e713520b346a38130779619a3fd58", "b9817ce61ede429f92856d3099741e7c", "e818445ff9ef415eb363b531dd9a11e1", "facc965d0df24a2fb4f60cd638e08c12", "0f0c2131a36447b2a7d951643a363df6", "bcba953ead0d4536bb74b6c4ff3c74d6", "e63af64276474e5f9102cde7acb5e43d", "3029e53ed1714ac78deb1ea012a151f0", "d2242894d29746a09e8850594189ea19", "21a245a014d0487bbb8e36809c9f8ec4", "dd93daf9c1aa40609245a2620a473bff", "51c1f000d4bd4497bc010f38083a8cb2", "aa63385adc974dd19ed07870a01f7c68", "7e7f48b74e364b1dbc91ba8f4d9b56d0", "e3974bd576ba42df80870c8468e293e8", "882c541bdb184af8a6e580fbbbcdec82", "3920fabf57904e51a25e7908fa101f5b", "426df53f13a24cd894197b1533b38c22", "9fb3c580ba274cb5a6facd25b9a5104f", "14d4de83c85f4e4ca9dddcf8fe3fc7b5", "f03854fbf8074d078708b7a264a598f9", "ab51bbb0b37c4b2da36628efdad7a263", "b6551400a66c4c339be1bd55656bdcfa", "2e005caa953144db92f90be12aa2ba8e", "ffddcad9659345f5b3658e171105b137", "64b16b9b25d1478dabb9edc5b13f4c38", "83c6f23406014e3bb6595c1b4af08c22", "cc9856e90f804c539aad305ead0bc752", "0d9140b2a82a41fca1e157415504e998", "86dedf1ffbcf4b128064529060cb1ae6", "1eef70fe9a7f4a39bff5545320f5f56f", "58812774d4c04de4a61e12168da71b17", "b324eb7f37884c71aa39afd411978345", "1f0a1d06ee624ed6a6a3bd0dba8c7d7f", "cc86abf2d3884b87ac64bcf760763881", "ed40a83450e44484b4f57806d6d87246", "84b61379ec7845418e31d586a944009e", "e1519af9225444c3981d5049f5275040", "8add11bc34984617a1c82f4f5665aaf9", "2849e175643a4508a923a3d48d91e1e6", "6f7df5ffdeab4d66b9ae621e7387fdf9", "1be9df4aacd748f18d523c289c53397c", "8ca1bf9a6055413c8de97fba969de4dc", "8f4b7f2e98184620a3a7ea85cbeb321a", "5c230ba8a98e4fc38ae15eefbe0375e8", "93ba6246872540df83c2d513568f7aba", "4d238d2d82ac4dec9b6bc53d4d1166ac", "aa0819f9c1cc49248b6499733f12eb79", "0fab4f04206d43a2bf0b257d045bd728", "a15d327645f64f22b5b535522b56abdb", "436cbf07ae36436f9ef052226f2c40bb", "7d06726eba114e1e96eddc93a24ed010", "474114a9dff6493fbb12429200813c39", "9c76070baf63494382b649bbdaa47826", "565e240e16154f51839c73e9e50a39b0", "68910a1920a041d9b7d745271bedb0b4", "ed475322addd4bb5b2b4aa08f691d26a", "285544641d6e429396cb3fac1227e53b", "09f8201aad994d7bb217d0c41fa0f259", "7a11c402406a4d83bb886cb5e7a9d4fa", "0b8b2da541e64dfbaeaf15bc0ac4c95f", "f87556a5ccf14226bfe17cd6f4aa503d", "b8f5826e36dc46bf8e19d12b3a3582f3", "64a88769952f430eb70a45966ffd9c5d", "bc6416e1a3c9471198daee5cf9a01778", "9407dea679554aa8801be3d4f2cfac3d", "5c022f716d204e43a63074197e74b4f9", "f84a5d35ef05426abe8cf6cca6b31b72", "0b9d672d9dda4058a64a8adc4cd0471c", "6c0d6ca97d6b4d569fd9dc22079c67c1", "2b5fb88c97bb4c5e90ce98af27c51f8c", "1851a7f697b54bbab424b4a92a0c2933", "01196d8ecd2a4bcb810fc5b2834651ac", "98f91c8af4ed474fa74e2b46e9ed7c51", "1999c3b20c82460d8f4fcd2194b12e18", "1b7747380b0a40009c74f80a2755a2ac", "5ccd63e6982646b081ce9eb8936d6b8f", "adb2d2a2f81a4f2fbb6506869c3151cf", "b0f64fd1d3354df58283086d22799b8f", "25d2a6e47bf940a19cd1afe64d60e3f3", "f4025a926daa4f979e026aa50b4ca893", "8f597829b004422a9216c827e7d2ab7a", "5799bdf83e1f45f08c1b0dc2128d9797", "9f71a2f25fc84882afc6109ca8f2806c", "fb894d61dd66418f8e189ad7cd0590e6", "e45d3dd199a048bca54bd843556900b2", "3422899483af4ce1831eea2f85a2d1dd", "a09c3fa4db71458e913da7d9e8f89fd5", "fdf9daf8583643c49516aa561dbd355c", "674266a4c23b4a85b8f929477494e696", "69aee22a44c74f1b973b98518708265e", "690429d61b5b4bf68ed5f88c41170082", "3554e1f30835413d87ed57039c77cb01", "0abcca7e098a4bb29485cd30b0fbece9", "90aee6c6cf3c4a6d8e38aecbfc7bd442", "f198806252ee47008d8ede24ab05d36f", "3833cb447d704beca06e42b47ce8ef25", "342494eeb78846b286d883f8ecdab46d", "043d9a17966349be88f0c0530064015f", "4391e0346b244e0894857297dc38365e", "c3be88cc5b124928a0510b42f4620e5a", "9dda911ba6fc4feeb6df8d9ade69db17", "93a316edf862485f9b0898a8ba6a1b03", "46a0c5ba67b9405fabb0bb05b2ec3aac", "27abc1442415407e844b0f41cba401e5", "7a1e2cd794e54a33895a74708aa0e4dd", "7cc14c43ccdb42589322a9de1291c9ae", "fcea2c37e4ad45b7949e6e4cbbef4ccf"]} id="ZpsocgBQr88m" outputId="17e31d21-87c9-40d2-d191-7024c7f526a5"
Dagger_Lander = Lander_Net()
device = "cpu"
loader_args = dict(batch_size=128, shuffle=True)

train_args = dict(epochs=100, lr=0.001)

env = gym.make('LunarLander-v2', new_step_api=True)
DAgger(env, expert_lander, Dagger_Lander)

# %% colab={"base_uri": "https://localhost:8080/", "height": 628} id="eQpcV3wSSoPk" outputId="c94a13d9-bccf-4c40-ad03-e7a36349354a"
anim = generate_video(env, Dagger_Lander, 500)
HTML(anim.to_html5_video())

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["24f8fd6b168e4d1aaf31859e9c7415c6", "5d88839dcfb04240909f24af5f2abf05", "73442eacc5bc482e9325129b3c7180b3", "7288b097dca34d20a0c6bdbedde091c5", "ca9e32f2b9e840299487e299d0628fe0", "ec4a14b7556d41ea88e41a9f530ec0c1", "f674b6cad32e4931b18c316c800c1226", "42e8497c243c45be9d408d5200445602", "6d67818dfcb74f849ad3ce68af646486", "0e70cdbf998b453397ebb117004ea147", "e7a36c18fb484a60b663c68da51e102a", "2bc987d5b2664ccd805ccb98825fd178", "bcbb4a62d1cf4a3abd77773699843931", "318b0932effc4539a76df94583d5d0ec", "2ba7b7e9f8ac4299895bf326b0fd3ec5", "74aaf3a4d9c74a2387751b2bcfbcb1b5", "0358d4749cdb45f7bfef16bcf7d9159e", "24ec99fc4209452ca0ac28754f0b17bb", "8f2aa54948d44567a8e37d725e1fed55", "fcf9f8c7ef4c4296b0fe1f6686fd998a", "29f6bc100b0e4bca96b2a59b59dbc38c", "3786701fb3f04576bcf79518da5d00a1", "c1c2fbc9c77846fd92bad250cd330263", "5dc5aa5883b24a2a93a858d064a9e5c4", "44e2871c0b9b48c3841a3dfbd0aed695", "623eb8956cac40daa52600194d00bad8", "d297de9adf5941298ba26e1d6c6d21d9", "d411a89a5e084ef6be4c5be8f553cd32", "5f011f33137f40ce95d8c73b001c599e", "6a5eb3e191af49848f8dcae5e4547f0d", "a6dda790104e4f5180a5fef161760a16", "64c262f15427404fab7208853c9c2ce1", "9b5436d343a8449480dec2bce0714191", "941e70f887e54f468e5c04c8efe36443", "2abd3f0e4cb143928e000e169e7705df", "940c58b303cf4465a9a739067c4ae6a5", "22fd6905ccfe446ca752c13b73bcf270", "a7203e2e66844f5dad4b5c9d9354f4f5", "3ce5f50f5c0144438a42141cfea1ec6c", "abc18023f53f4e39a1cfbd1f80c8ee51", "3499fa0aefaf4cb88cf6f12c2109f822", "1c26684482b647438b1b48c3ef1d93d1", "da7f3c82489a48eeade663410bd82531", "2cb31b9cb6dc47428cc256ede82c6d40", "5b30e0aa2cfb4d54b0626a9a7c3f53a8", "d331cfaff8f54af18773f167ae77963f", "124a6cfa8fba4be39068bb712d07565c", "d07f89a67fad466186552a2a1b1e4cfa", "12e9a7b3a00943de999612dfbda467b9", "699714024cc84243818e9138ac1922a9", "6b4bb6a337dc40adbf8a944806695daf", "110959069e314b9297885b40271e0cb7", "09c601acc5aa49558b405bbeb738a47d", "54bc48e8fe7144ed90dc6a6556349b52", "1545cf16b8404f26ac64d64a5461eff7", "e8189ae84de54b7794b708f22364d308", "bc4f852dd88d4951b95e32c2f3568d75", "efd890f4069c421395bb793fc700a2c8", "4091092a166f415980f1dfed002ad47e", "012b0f67f9564e97aaa16dd63e933fff", "d55d8d052dbd42dfb824b20c7deaaed6", "9da91ae23c274f4cb3bbdc533f708661", "7a0464d8a239468e9734de83604620d1", "af476296ee584f6fb184de7995aadc1f", "d233d222681c4f98bf7ad33ef5636b99", "e96de680a8b44a468bebb97ab9513266", "f4802bb89af9454ab2f0667578a11ad2", "16ff45e6cb03401fa22c80afe3c4e458", "05bc375f98ca4662b2e1c957a33a3aa1", "8d74dd6a6d124f0a9da5952f7bf5da13", "69fe4fa6ff67403daad11441db563562", "2278a226d5af4280a07205b5b9e86563", "391fcf55985e4c1dbce3f46d1b4742f9", "3cadb8501fbe4404a9525f2986afafdd", "e6671eed6fbc48b4acacc74ed9e3833a", "d150d608e3ef40faab96ddaf9b2b916d", "049937b10ef040d7ba35fe730f452a9b", "e2ab7a2e76124aeeb76d8648fbc3d71c", "45a45028a1e240699c41dca9fa5b12f6", "c666c193a5184841958193c8a417332d", "fec5ee7085ac48e9aedfbee06e0922d6", "0ea91467167b44f5bc2b8451b96eb2b9", "1f7f6aa55eb543a697eccba3723c5882", "034ebf7c5e9747d887632e1b34fc9f7e", "ba9b6f5033e8475191f471740bdafaaf", "9b5af2e73847412fa6711d79a34954cb", "32cb7b031af8466f99aa8f5eab491f47", "f8b5ea098cda4f208828929ace7d80c7", "11c936b3cf614e82a545bd75efe2f0ea", "9f2e5e3ba24a4747abdb9486feac8b07", "894d6db374c34c398a474365a8c15dc7", "8ad53749b1934620a6b7f43b8fcd05fc", "6ef9f0fb9ba64541a69d280d00074a18", "54a823f9b2a84e2290562fd4c8f83322", "d7ef53ce466c48e1a6ff1e7ef18d4b53", "e94a0a1974134ea18a06dea43e38d4f2", "b774abe4966a4bc6ad7ba3c9e1f6d55c", "6c657cefa8a14af08285522eab52b182", "01b1a1563264456d9980a22a4f3c79d7", "1f292a2cd9354ea7a58e961ab456e086", "df6061d892c24d53a3c9f395e4afe9a6", "716a39c189174c4d9ba537107d633389", "6c432757136e49158bcd52f78696373c", "cea0da9ff0394fb8bd0451f4bf9bb747", "59ddb3e500974b6399a6056d512ff3be", "9691ca9cc7064752ae6d6d9e22d9a799", "b5857c3fa79e47079cd4d141d96d942f", "55f39671f3eb438faea122d2a09cf514", "c20d194786c249e493069040308ecfbf", "46f670959f364b5bac08cd787a0a0d5e", "22f048a2f53c4752b2efdacb5a53dc59", "479cf6c15ea8408185be94a296bebce2", "43bd98136bfa4ca9ba972e3206aad3cb", "3d6c9536a0a4454db69533d2a165068f", "183dd570998a4e13bb77c5e7b4a50120", "9fbac82da4194b289cf71708b1d91d19", "ce43741d077646d08135b63d19149e75", "cff0914c42e34bac88ede5a929396436", "a0126f7baba448de9cfa32960432abbf", "c1d03c3b688f491ba68e3501c22c045f", "9cddd0c81a534f2e8079bba90e6da49d", "3f8be6cd707c4d04a9886e5c9039d83b", "b908e923aea84ccf98dca812961083b6", "a35756b1e66d4aee86f286d7154f71e4", "a7539ccbb62b4aa18226c097419d5364", "bd855dca52a54e5980ecfaff467bd139", "a0b791d9583843ac90279945fe41875c", "449c221727c8482486862c5bd97e298c", "e9ae1d2feee242908b15059ba2336fdb", "6d805616ab6d43089f643a0c33fac88a", "3ba58b8cafde4328b1d48a36ae5fef5e", "c14c2eefccdf4029ae9957b56ca0ed12", "6cf074474ec24689a651e5eb9e26d69e", "8647f6987d524189a078aac87e5ba38c", "65b0866c98fb4a97816283b44a1d9613", "7b22d1251e424169869f25179ab658e1", "ba5a7131940646b9b0592bf198f27ae7", "a8df7cc9cd054ddfafe9206bb670ab36", "e20e47bc8e7b420290ded6af059e12b2", "5226a46c2b374f40bd411c5a30decfa6", "d5abeb56be944b19a7ed19405d71148c", "d97c1c42003841ca967544271b3732a6", "dfe432eb7b7d44bd82471eb6afafc662", "8093c13b4f7949e9b8ba367ae0175b1a", "b9dae37179734a17b746f32c4c02b28d", "1a243e891ff243bd9659cbcf53a6b029", "d02dfbf2650c40cc8ca2edf18f9afa7a", "46797c802da64fc49448237034790638", "2f0314cda1f5466aa508672b06fd15fd", "abb766db576b4cd2a7b6ecf33721c254", "d608a120dc7e42b8bc5e08015c03ce2c", "e01463de48b54c96ab3eef0380f7f018", "cee306c6a03a41d2be6d4d8d86402d2b", "7a654eb97d1f4f32bac392d0d30ea84c", "123bb222702f4677a124c42b1a912434", "5eec0dd59437404f9202f787389bbe31", "944efb864c8f454ca1c0b2632754d8bb", "63b3c6541131448baa9e5ed43bad27de", "71bf6d0d00f44c008ea2757552fef437", "41db1284a7d14eeca9dde22c6321a058", "97eaf2803ce34a8aa8c4753a1dd45d9f", "d39fd9d6309e4eb6846f79cced362a29", "0dfab02e8fa641058de1c507ecb0fe2b", "28a7c19484bb45a283e8c970abd9af11", "b5be71d4ce34417bba94128de7eb5b8c", "07ebba51d2bb47b191c9552055926cef", "2b483e54620248cda501d66b3bf712ea", "05316389cd684b259f5e477cedc72a82", "5022dd8c64fc44aa915818a5deece9af", "b577965fed8e4d518c919ff2be96f2f2", "9b1650f9ea3a4cc2b46d7e2a4c90e957", "8efd206dbc9347f587468b1d93d929d3", "c4601dffa6174182b1138a637f02efa2", "43325ec3516c4f3490c51e9bad2229b9", "3a3f28d14675442cae4a6e174a31c616", "4ab4396f65a74012939654e068b4ed41", "0e2d7c43220f47eabcc12fd566479941", "a0a9bd6acb38422e80f7844f44bc4bb8", "7c258dfc7a8744e082f4b24b98f0e4b2", "c391a1f7a6684318ab4b67b7aaf2aa54", "f05d3be33a2f4f169371eaac89b3897c", "22c21a1d44b04f4aa5391494d7201521", "008edc240bda437bb2bb3526fc1e8fc4", "815cbe391900412f9098ba48894d0c2f", "492241f3cadb49ff9ba5252ea8c44672", "b257c7a4682d4d5392a33377d9af7b5e", "7f04b5fb234c450c94e6a7dfa1f7598a", "f08a7dff2c0845ac91c0074b151214f0", "63175be285454b178c54623e110d8f48", "7e8f1b9b9c26402d9296b4e198ef4c06", "fffba30dd6794b71ba6a8f8bfa1ffe91", "7465251e2c54483ca32177a9a82318b8", "542be908ea1f42d7adb51a2ed1c8daff", "a98459a77ef045019b2da556f6dcc1e3", "fe4da51b35dd42f4992b98632c87582a", "957f6e10c12a4f7086c86b7489a257cc", "279ac15649d54b7ea39d1a75b803ef82", "8ec9012632794211b5fbb858ddd6f880", "4e5279fccabf4f13bbdca9029e4e8630", "9c80aa3f52e044aea50c4e0b596bf041", "a001f886bd244b7988a1e547fa55375f", "52158157fe49446fb55c8f36163221f8", "d17b30a27b464713b33d39207e5e42e3", "cc6e64fca21b46fcab9a0b7a94f28b03", "77a2074fddbc41628d0678fa78a50cc6", "597699521c7f46a182e3fe05d48e5e91", "8dbfdd2749d944989c29b26e168f350e", "4ab608d12d1a4ae6b579991b02ea5d64", "23f3bef6dc7a4865b7226939b285ac8c", "abe1aec96b5d4bf58a1fc4a50e44c639", "b66ab2d8a8ef4c4e95d3325389c57943", "c228aea2fb55471e85793ecba01b6405", "44b05121ace5431aae3e0a33a8a7ccf2", "04f6e56591c4423b90c34141e30b3a25", "0c78744c42f944fe95daa9c46a83b782", "a73ab713bdf0467a8bccf2a2b6f80182", "b7001a8dcbaf456f8a74faf2e637eb1e", "12e408906a334bf2aaae6f49460017ad", "8d120a9c62aa4cf4a9cd64a425147e02", "eb08caf70f744221bb3213bf37af9cff", "52f8ddf8ffd34538a4a7b92cc7b1a33b", "2f97c81bc9cc435ea3feeba32276f911", "fb102e931da74269b2899520a0af82c3", "7e1a13f7cee344379aa66677708fee2c", "a251417440534c20a1e145b6562a233c", "f6335b09ef85409ca1cd454978658285", "9d0507bfea734185905df08be797cf9e", "53cd391d88044f3fb5aac6c42ee4ad0e", "a71bb6516b0745bfbe2eecb01c997b26", "370655b21a22479e91fe0a92391f4fe3", "bf2caeb0a136422ba16ef9c2179e6c3e"]} id="QBBbxpNJTAL4" outputId="00806347-58e6-471a-aeeb-2ceb1eaf3d19"
Windy_Dagger_Lander = Lander_Net()
device = "cpu"
loader_args = dict(batch_size=128, shuffle=True)
train_args = dict(epochs=100, lr=0.001)

DAgger(env, expert_windy_lander, Windy_Dagger_Lander, n_iterations = 10, n_traj=50, iter_plot=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 628} id="2fvk4d3TTgbQ" outputId="dcfbfa2d-3d07-41ee-d683-d142d772c133"
anim = generate_video(env, Windy_Dagger_Lander, 500)
HTML(anim.to_html5_video())

# %% [markdown] id="vRndNh6TMiUT"
# ## Part 3: MaxEnt IRL (bonus)

# %% [markdown] id="Ft_yIpZ52sVu"
# It would be great if we could avoid having the expert in the loop while making sure that different initial states or stochastic dynamics have a limited impact on the agent performance. That is the aim of Inverse Reinforcement Learning. Instead of learning a policy given a reward, we learn a reward given expert demonstrations. The aim is to retrieve the reward function that the expert followed and then use that learned reward to train a classic RL algorithm.

# %% [markdown] id="MIXH0id13UN8"
# Below is the *Maximum Entropy Inverse Reinforcement Learning* pseudocode ([B. D. Ziebart, A. Maas, J. A. Bagnell, and A. K. Dey, *Maximum Entropy Inverse Reinforcement Learning*, in AAAI Conference on Artificial Intelligence (AAAI), 2008](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf)).
#
# <b>Initialise</b>:<br>
# - feature_matrix as an identity matrix of dimension number_of_states <br>
# - expert feature expectations as a vector of size number_of_states
# - environment to get initial state $s$<br>
# - q_table $Q$<br>
# - $\theta$ and learner_feature_expectations as vectors of size number_of_states ($\theta$ values are negative)<br>
# - $\gamma$ the discount factor
# - learning_rate
# - update_freq: frequency of reward update
# <b>FOR</b> episode in number_episodes: <br>
# $\quad$ <b> IF </b> episode % update==0: <br>
# $\quad \quad$ gradient = expert_feature_expectations - learner_feature_expectations/episode <br>
# $\quad \quad$ $\theta$ += learning_rate*gradient <br>
# $\quad \quad$ clip $\theta$: all values that are greater to 0 are fixed to 0 <br>
# $\quad$ <b> While</b> True: <br>
# $\quad \quad$ Draw $a\sim\mathcal{U}[0,1]$ <br>
# $\quad \quad$ <b>IF</b>: $a<\epsilon$ <br>
# $\quad \quad\quad$ Draw action randomly for action space <br>
# $\quad \quad$ <b> ELSE </b>: action = $\arg\max_a$ q_table[$s$] <br>
# $\quad \quad$ next_state = env(action) <br>
# $\quad \quad$ reward = $\theta$.feature_matrix[next_state]$^T$ <br>
# $\quad \quad$ $Q(s,action) = reward+\gamma \max_aQ(next state,a)$ <br>
# $\quad \quad$  learner_feature_expectations += feature_matrix(state) <br>
# $\quad \quad$ <b>IF</b> done: break
#
# You can find additional information about *Maximum Entropy Inverse Reinforcement Learning* in section 18.5 of *Algorithms for decision making* by M.J. Kochenderfer, T.A. Wheeler and K.H. Wray at MIT press (freely available online at https://algorithmsbook.com/files/dm.pdf).

# %% [markdown] id="DMOsvpQKsFZ3"
# Let's test with the Mountain Car environment. The documentation of this environment is available here: https://gymnasium.farama.org/environments/classic_control/mountain_car/.

# %% colab={"base_uri": "https://localhost:8080/"} id="dO91_TIwPZ6S" outputId="22125369-fceb-4577-de88-24e1d6c340a2"
env = gym.make("MountainCar-v0")

# %% [markdown] id="IBrUMqylsYl1"
# You can either train the expert model or load a pretrained model.
#
# Uncomment the following lines if you want to train the expert model (should take 10 min max):

# %% id="yPFQdNuQsYOP"
# model = PPO(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=250000)
# model.save("lab4_mountain_car.zip")

# %% [markdown] id="16L3SwcDsnPL"
# Otherwise, uncomment the following cell to load a pretrained expert model:

# %% colab={"base_uri": "https://localhost:8080/"} id="MRe_-A-0RbuJ" outputId="1a51198e-66b1-4a91-8a41-125134e85e93"
# #!wget https://github.com/jeremiedecock/polytechnique-inf639-2023-students/raw/master/lab4_mountain_car.zip

# %% colab={"base_uri": "https://localhost:8080/"} id="KTGVowHhRkaM" outputId="2c056e2b-e805-435e-c70b-4b0560244298"
expert_car = PPO.load("lab4_mountain_car.zip")

# %% colab={"base_uri": "https://localhost:8080/", "height": 593} id="h5tzzh2dRtDk" outputId="3cdbe0f2-bb12-432d-9599-8ffd8cc54f12"
env = gym.make("MountainCar-v0", new_step_api=True)
anim = generate_video(env, expert_car, 400)
HTML(anim.to_html5_video())

# %% colab={"base_uri": "https://localhost:8080/", "height": 192, "referenced_widgets": ["c99c555d90bf4545b0037f11b327793f", "883e5888e7cc46a2b806e790a56cdb34", "54769c3e4c284f80a759609031b29b61", "227cc47199dc41969b2cc41f94315bd3", "e825105aeea14d759e140d98288ba9ee", "becb1a2e49d64a4d9aa046bd87c1a6e6", "8647180fdf4e42d5b4e44a27d6b832f1", "09f02a13e1924bfb82192d33c6534281", "1d188a7a23324c0c8f095edf85e1d790", "032c35c835d84db59abc1566e13f23c9", "065afeb2bf544fdfa8e04113aa9da9f9"]} id="1krNuiUIS9od" outputId="ef65cf7a-efe3-4f88-d4f5-d71cb9dda25c"
car_demos = generate_expert_traj(expert_car, env)

# %% id="gZ0tC2bIVtJO"
# The state space of mountain car is continous. As we use Q-learning we need finite state spaces. We therefore discretize it
# The state space will become an array n_bins*n_bins (one dimension per state dimension)
n_actions = env.action_space.n
n_bins = 30
n_states = n_bins**2
feature_matrix = np.eye((n_states))

def feature_vector(state: np.ndarray, env: gym.Env = env, bins: int = n_bins) -> Tuple[np.ndarray, int]:
    """
    For each state, this function associates it to the corresponding state index and builds the feature vector.
    The feature vector is a vector of shape n_bins with a 1 at the state index.

    Parameters
    ----------
    state : np.ndarray
        The state to convert into a feature vector.
    env : gym.Env, optional
        The environment the state belongs to, by default env.
    bins : int, optional
        The number of bins to use for discretization, by default n_bins.

    Returns
    -------
    Tuple[np.ndarray, int]
        The feature vector and the vector id.
    """
    try:
        idx = np.array(np.arange(bins**2)).reshape(bins,bins)
        bin_length = (env.observation_space.high - env.observation_space.low)/(bins-1)
        bin_id = (state - env.observation_space.low)/bin_length
        vector_id = idx[int(bin_id[0]), int(bin_id[1])]
    except:
        print(state, env.observation_space.high, env.observation_space.low,  bin_length, bin_id)
    return feature_matrix[vector_id], vector_id


def process_demos(demos: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
    """
    Classify the state-action pairs that belong to the same trajectory.

    Parameters
    ----------
    demos : Dict[str, np.ndarray]
        The demonstrations containing state-action pairs.

    Returns
    -------
    Dict[str, List[np.ndarray]]
        The demonstrations classified by trajectory.
    """
    demonstrations = {}
    demonstrations['obs'] = []
    demonstrations["actions"] = []
    new_episodes_idx = np.where(demos["episode_starts"]==True)[0]
    for i in range(new_episodes_idx.shape[0]-1):
        demonstrations['obs'].append(demos['obs'][new_episodes_idx[i]:new_episodes_idx[i+1]])
        demonstrations['actions'].append(demos['actions'][new_episodes_idx[i]:new_episodes_idx[i+1]])
    return demonstrations


def expert_feature_expectations(feature_matrix: np.ndarray, demonstrations: Dict[str, List[np.ndarray]], n_states: int = n_states) -> np.ndarray:
    """
    Build expert_feature_expectations from demonstrations.
    Feature expectation is a matrix of size (n_demonstrations, n_states), each line is the sum of all feature_vector for each state encountered by the expert
    in a given trajectory. More specifically, the feature vector of state 1 can be something like [1,0,0]. If the expert followed the trajectory:
    state1, state2, state1, state 3, then the feature expectation for this demonstration is [2,1,1]. Then we take the mean of the matrix for all demonstrations
    to get the expert_feature expectations.

    Parameters
    ----------
    feature_matrix : np.ndarray
        The feature matrix to use for building the feature expectations.
    demonstrations : Dict[str, List[np.ndarray]]
        The demonstrations to use for building the feature expectations.
    n_states : int, optional
        The number of states, by default n_states.

    Returns
    -------
    np.ndarray
        The expert feature expectations.
    """
    n_demonstrations = len(demonstrations['obs'])
    feature_expectations = np.zeros((n_demonstrations,n_states))

    #BEGIN ANSWER
    
    # TODO...

    feature_expectations = ... # TODO

    #END ANSWER

    return feature_expectations


# %% colab={"base_uri": "https://localhost:8080/"} id="fnzdg-wbPliT" outputId="64841cac-4455-4fd3-b2df-e192d6006065"
demonstrations = process_demos(car_demos)
expert_expectations = expert_feature_expectations(feature_matrix, demonstrations)
print(expert_expectations.shape)


# %% id="mMsRUuK91q1S"
def maxent_irl(expert: np.ndarray, learner: np.ndarray, theta: np.ndarray, learning_rate: float) -> np.ndarray:
    """
    This function calculates the gradient and updates theta.

    Parameters
    ----------
    expert : np.ndarray
        The expert feature expectations.
    learner : np.ndarray
        The learner feature expectations.
    theta : np.ndarray
        The current theta values.
    learning_rate : float
        The learning rate for the theta update.

    Returns
    -------
    np.ndarray
        The updated theta values.
    """
    #BEGIN ANSWER
    
    # TODO...

    #END ANSWER

    return theta


def get_reward(state: np.ndarray, theta: np.ndarray) -> float:
    """
    Calculates the irl_reward of a given state.

    Parameters
    ----------
    state : np.ndarray
        The state for which to calculate the reward.
    theta : np.ndarray
        The current theta values.

    Returns
    -------
    float
        The calculated reward.
    """
    #BEGIN ANSWER
    irl_reward = ... # TODO
    #END ANSWER
    return irl_reward


def update_q_table(q_table: np.ndarray, state: int, action: int, reward: float, next_state: int) -> np.ndarray:
    """
    Performs Q-Learning update.

    Parameters
    ----------
    q_table : np.ndarray
        The current Q-table.
    state : int
        The current state.
    action : int
        The action taken.
    reward : float
        The reward received.
    next_state : int
        The state transitioned to.

    Returns
    -------
    np.ndarray
        The updated Q-table.
    """
    #BEGIN ANSWER

    # TODO...

    #END ANSWER
    return q_table


# %% colab={"base_uri": "https://localhost:8080/", "height": 607, "referenced_widgets": ["e6b3e6876bd6489f8f1dcdd70e3a3c3a", "b415891505ab4666970daf428df88138", "ca96f2c80b8e4a29b96aec30c43b7823", "f0f42af0321d4e4babdade8db655c3e2", "79b47a93d26147338e16edc78120dc2a", "e8cc46552d5c467eb462655ca1e2e5d7", "44d6a472fec84629aedb971d02230b3a", "29c3cb8c52934daeb4b4f251b3fb305c", "10789dddaa2745058b5f7e6961515b02", "b7fd676f32fb44bba28674fef78bf5ce", "3b9a6007bfbb46c98f83461ac6fcf124"]} id="av50OqwTMk0X" outputId="ce7cfa84-e545-4143-cbc3-c0746dc9d589"
env = gym.make("MountainCar-v0")
learner_feature_expectations = np.zeros(n_states)
theta = - np.random.uniform(size=(n_states,))
q_table = np.zeros((n_states, n_actions))
gamma = 0.99
q_learning_rate = 0.03
theta_learning_rate = 0.05
eps = 0.2

episodes, scores, mean_scores = [], [], []

for episode in tqdm_notebook(range(30000),desc="Episode", leave=True):
    state = env.reset()
    score = 0

    if episode !=0 and episode % 5000 == 0:
        learner = learner_feature_expectations / episode
        theta = maxent_irl(expert_expectations, learner, theta, theta_learning_rate)

    if episode !=0 and episode % 1000 == 0:
        plt.plot(mean_scores)
        plt.show()

    while True:
        _, state_id = feature_vector(state)
        a = np.random.uniform()
        if a < eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_id])

        next_state, reward, done, _ = env.step(action)
        irl_reward = get_reward(state, theta)
        _, next_state_id = feature_vector(next_state)
        q_table = update_q_table(q_table, state_id, action, irl_reward, next_state_id)
        learner_feature_expectations += feature_matrix[int(state_id)]
        score += reward
        state = next_state

        if done:
            scores.append(score)
            episodes.append(episode)
            mean_scores.append(np.mean(scores))
            break


# %% id="OURZR2I5ZTEP"
def generate_car_video(env: gym.Env, q_table: np.ndarray, n_timesteps: int = 300) -> matplotlib.animation.FuncAnimation:
    """
    Generate a video of the car's movement based on the Q-table.

    Parameters
    ----------
    env : gym.Env
        The environment in which the car is moving.
    q_table : np.ndarray
        The Q-table guiding the car's movements.
    n_timesteps : int, optional
        The number of timesteps to include in the video, by default 300.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object representing the video.
    """
    obs = env.reset()
    figure(figsize=(8, 6), dpi=80)

    # use False con Xvfb
    _display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
    _ = _display.start()

    frames = []
    for t in range(n_timesteps):
        state_id = feature_vector(obs)[1]
        action = np.argmax(q_table[state_id])
        obs, rewards, done, info = env.step(action)
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        time.sleep(.025)

    anim = plot_animation(frames)
    return anim


# %% id="ys-fhUAQQT0R"
anim = generate_car_video(env, q_table)
HTML(anim.to_html5_video())

# %% [markdown] id="KPEnmR_vOAhk"
# ## Further readings
#
# We were able to solve mountain car using our learned reward. A final remark is that each time the reward is updates, the MDP has to be solved with the new reward. Solving an RL problem is usually hard enough so solving it multiple times is nearly computationally  infeasible for environment that are bigger than mountain car.
#
# Getting rid of that constraint is one of the main research topic on this subject. Several approaches solved it. Those interested can have a look at:
# - [GAIL](https://arxiv.org/abs/1606.03476)
# - [Guided Cost Learning](https://arxiv.org/abs/1603.00448)
