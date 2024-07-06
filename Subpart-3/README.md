## Introduction to Reinforcement Learning Algorithms

A Robot Learning setup is primarily composed of two components, an agent and an environment. Then environment refers to the object that the agent is acting on, while the agent represents the RL algorithm. The environment starts by sending a state to the agent, which then based on its knowledge tries to take an action in response to that state. After that, the environment sends a pair of next state and reward back to the agent. The agent will update its knowledge with the reward returned by the environment to evaluate its last action. The loop keeps going on until the environment sends a terminal state, which results in termination or end of the episode.

### Types of RL Algorithms

**1. Model-based vs Model-free** : The model stands for the simulation of the dynamics of the environment. That is, the model learns the transition probability T(s1|(s0, a)) from the pair of current state s0 and action a to the next state s1. (This is the probability that given that the agent in a state s0, and it takes an action a0, then what is the probability that it ends up in state s1). If the transition probability is successfully learned, the agent will know how likely it is to enter a specific state, given current state and action.
If you know this transition probaility matrix, then you can easily use it to plan your policy.
   On the other hand, model-free algorithms rely on trial-and-error to update its knowledge. As a result, it does not require space to store all the combination of states and actions. You will be working on model-free algorithms in this track as model-based algorithms, tend to become impractical as the State and Action spaces grow, since learning the state transition probabilities becomes very difficult. Think of it in terms of robotics - If we consider a certain angle to be a state, then it can have infinite values since angle can be any real value in  a given range.

**2. Off Policy vs On policy** : An on-policy agent learns the value based on its current action(a) derived from the current policy, whereas its off-policy counter part learns it based on the action(a\*) obtained from another policy.

![image](https://user-images.githubusercontent.com/77875542/182362118-71b13eab-c1fb-4c38-bdc2-2016e4710d47.png)


### Q-Learning

Q-Learning is an off-policy model-free RL algorithm, which forms the building blocks for Deep Q Network. So, lets go thorugh some basic terminologies before diving deep into Q-Learning.
<br/>
**Value(V)**: It is the expected long-term return with discount from the current state, as opposed to the short-term reward R. Vπ(s) is the expected long-term return of the current state under policy π. It is a function of the current state only. It represents the average reward the agent can get from this state to the end of episode considering all actions.
<br/>
**Action Value or Q-value** : Q-value is a function of both the state and the current action a. Qπ(s, a) refers to the long-term return of the current state s, taking action a under policy π.
<br/>
**Discount(Gamma)** : This is a paramter having a value between 0 and 1, and basically decides whether the agent should try to maxiise the immediate reward(R) or the reward it collects in a long time.

Methods in Q-Learning family learn an approximator Q_theta(s,a) for the optimal action-value function, Q*(s,a). Typically they use an objective function based on the [Bellman equation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html). This optimization is almost always performed off-policy, which means that each update can use data collected at any point during training, regardless of how the agent was choosing to explore the environment when the data was obtained. The corresponding policy is obtained via the connection between Q* and pi*: the actions taken by the Q-learning agent are given by

![image](https://user-images.githubusercontent.com/77875542/182362335-fa0844ae-636c-4e17-aaad-ebe46ce86dc0.png)


### DQN
DQN(Deep Q-Network) is one of the most commonly used RL algorithms and uses the Q-update step to get to the optimal Action Value function. Since, the the number of states and state transitions are extremely large, even in simple environements we use a deep neural network as a function approximator in Deep Q-Network. This neural network learns a very good approximation of the actual Q-function by interacting with the environment and getting performing the Q update at every step as mentioned below:

![image](https://user-images.githubusercontent.com/77875542/182362291-cfe0a896-1b48-4ed4-8e51-dc6520d93414.png)


Since we are using neural networks as approximators, and the target network is same as the learning network and is getting constatntly updated, this might lead to unstable learning at times, so DQN has some variants which tackle these problems by introducing the following improvements:
* **Target Network** : Use a different target network, that gets updated less frequently as compared to the learning network, i.e. set the target network paramters equal to the learning network parameters every 100(say) steps. This is done in Double DQN. 
* **Experience Replay** :  Experience Replay stores experiences including state transitions, rewards and actions, which are necessary data to perform Q learning, and makes mini-batches to update neural networks. This reduces correlations between experiences as they are sampled from the experience replay and increases learning speed.
* **Clipping Rewards** : Clipping large rewards and mapping them to a normalized value between +1 and -1 has often lead to faster and more stable training.

![image](https://user-images.githubusercontent.com/77875542/182362203-b20edaa1-ab84-493b-886f-045bb608faa2.png)

## Task1: Creating a custom robot environment in gym

This task includes creating a full fleged OpenAI gym environment, which can be directly installed as a PIP package from the terminal. OpenAI gym already has a lot of environments ranging from classic control ones to Atari games like Breakout, Pacman, Pong etc. However, you will be creating your own environment from scratch and training a robot in that using the algorithms learnt this week.

Before diving into the environment description, lets see the basic components in any gym enironment. So, every gym environment has the following file structure:

```
gym-robotics/
  setup.py
  gym_robot/
    __init__.py
    envs/
      __init__.py
      robot_env.py
```
Note: Here the names can be anything you which but make sure that you use consistent names in the files given below, otherwise it will give an error.

`setup.py` contains our environment details like name,version and its dependencies:

```
from setuptools import setup

setup(name='gym_name',
      version='0.0.1',
      install_requires=['gym', 'pybullet']
)
```

`gym_robot/__init__.py` contains the package registry information given below:

```
from gym.envs.registration import register

register(id='packagename-v0', entry_point='gym_robot.envs:RoBots')
```
`envs/__init__.py` will contain the following import statement:
```
from gym_robot.envs.robot_env import Robots
```

The main environment is defined in `envs/robot_env.py`.

### Environment Description

You will be creating an environment for training a 2 wheeled bot, known as Turtle Bot to balance for as long as possible without falling. You need to formulate this problem as Markov Decision Process and make an environment to train your bot. Tinker with the urdf, see what commands you give and what happens, and then accordingly try to build an environment which can make the robot self balance on its own. Here, you yourself don't have to code how to self balance the bot. You just have to define the state space, action space, reward, and the functions given below, so that on, seeing all this the algorithm can train your bot.

Below is the starter code, stating the methods that shoud be there in your env and loads the turtle bot URDF provided:

```
import gym
from gym import error, spaces, utils
import pybullet as p
import pybullet_data

class RoBots(gym.Env):

  def __init__(self):
    self.physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    ...

  def step(self, action):
    ...

  def reset(self):
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(0.01)
    planeId = p.loadURDF("plane.urdf")
    cubeStartPos = [0, 0, 0.001]
    cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    path = os.path.abspath(os.path.dirname(__file__))
    self.botId = p.loadURDF(os.path.join(path, "balancebot_simple.urdf"), cubeStartPos, cubeStartOrientation)
    ...

  def render(self, mode='human'):
    pass
```

The main task of creating the environment can be further broken down as:

- Defining an appropriate observation space that provides the information required to make a decision(take a certain action). Hint: The robot's orientation can be a crucial factor to determine its state for balancing.
- Defining an action space for the robot
- Defining a reward function, that gives the robot appropriate feedback of its performance and helps it acheive the target behaviour of balancing. Make it an episodic task with each episode lasting 200s.
- Implementing the actions in your action space, using `p.setJointMotorControl2(...)`

After creating you environment, go to the main directory of the environment and run the following command in the terminal to install the package:
~~~
pip install -e package_name
~~~

The last stage involves testing out your environment using the `gym_test.py` which would send random actions to the agent, so that you can observe its behaviour and the reward. You can modify the file based on your action space so that its sends legitimate actions to the robot.

# Implementation of Robot Learning

As you must have realized by now, implementing complex algorithms can be quite a pain sometimes. It just leaves you wondering why there aren't libraries like [Keras](https://keras.io/), [Tensorflow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/docs/stable/index.html) in regular Machine Learning, which make your lives so much simpler by abstracting away the nitty-gritties of implementation and lets you focus purely on the big picture. Fortunately, there are such libraries available as open-source, one of which is the [**Stable Baselines**](https://stable-baselines3.readthedocs.io/en/master/) Package.

Stable Baselines is a godsend for people like us who work on RL problems quite frequently simply because it works so well across multiple environments, and is quite easy to get up and running. You may think of it as the equivalent of Keras in Reinforcement Learning.

All the popular RL algorithms like DQN, DDPG, PPO, TRPO and quite a few more are provided off the bat in this package, and it is quite simple to use as well.

The docs explain the package so much better than we ever could, so we encourage you to have a look [here](https://stable-baselines.readthedocs.io/en/master/).

## Task 2:

Your task, should you choose to accept it, is to implement a control algorithm for the environment that you created in Week 2. We are aware that coding an algorithm from scratch will be difficult, therefore we encourage you to use the full power of Stable Baselines. You can implement any algorithm of your choice, but since you are familiar with DQNs, you can start there.

Make sure you use the policy of iterate and improve and keep tuning your hyperparameters to get the best possible result.

In conclusion, we must say that it was a pleasure to be a part of creating this camp, and we hope to see you spread your wings across new frontiers using this newfound knowledge.

We hope this is how you feel after this track of the camp.

![image](https://user-images.githubusercontent.com/77875542/182361793-a046ae64-5903-4d8a-bb93-c4a0dad3992e.png)

