# 基本用法

## 什么是强化学习？

在深入了解Gymnasium之前，先来理解我们想要实现的目标。强化学习就像是通过试错来教学——代理通过尝试不同的动作，接受反馈（奖励），并逐渐改善其行为。可以把它想象成用零食训练宠物，通过练习学会骑自行车，或者通过反复玩视频游戏来掌握技巧。

关键的洞察是，我们并不会直接告诉代理该做什么。相反，我们创建一个环境，让它可以安全地进行实验，并从自己行为的后果中学习。

## 为什么选择Gymnasium？

```{eval-rst}
.. py:currentmodule:: gymnasium

无论您是想训练一个代理来玩游戏、控制机器人，还是优化交易策略，Gymnasium 都为您提供了构建和测试想法的工具。
Gymnasium 的核心是为所有单代理强化学习环境提供一个 API（应用程序编程接口），并实现了常见环境的实现：cartpole、pendulum、mountain-car、mujoco、atari 等。此页面将概述如何使用 Gymnasium 的基础知识，包括其四个关键函数：:meth:`make`、:meth:`Env.reset`、:meth:`Env.step` 和 :meth:`Env.render`。

Gymnasium 的核心是 :class:`Env`，它是一个高级 Python 类，代表强化学习理论中的马尔可夫决策过程（MDP）（注意：这不是一个完美的重构，缺少 MDP 的几个组件）。该类为用户提供了启动新剧集、执行动作和可视化代理当前状态的能力。除了 :class:`Env` 之外，提供了 :class:`Wrapper` 来帮助增强/修改环境，特别是代理的观察、奖励和执行的动作。
```

## 初始化环境

```{eval-rst}
.. py:currentmodule:: gymnasium

在 Gymnasium 中初始化环境非常简单，可以通过 :meth:`make` 函数完成：
```

```python
import gymnasium as gym

# 创建一个适合初学者的简单环境
env = gym.make('CartPole-v1')

# CartPole 环境：在移动的小车上平衡一个杆
# - 简单但不平凡
# - 快速训练
# - 清晰的成功/失败标准
```

```{eval-rst}
.. py:currentmodule:: gymnasium

此函数将返回一个 :class:`Env`，供用户进行交互。要查看您可以创建的所有环境，可以使用 :meth:`pprint_registry`。此外， :meth:`make` 提供了许多额外的参数，用于向环境指定关键词、添加更多或更少的包装器等。有关更多信息，请参见 :meth:`make`。
```

## 理解代理-环境循环

在强化学习中，经典的“代理-环境循环”如图所示，代表了强化学习中的学习过程。它比看起来要简单：

1. **代理观察** 当前的情况（例如查看游戏画面）
2. **代理选择一个动作** 基于它所看到的情况（例如按下一个按钮）
3. **环境响应** 以新的情况和奖励（游戏状态变化，分数更新）
4. **重复** 直到剧集结束

这看起来可能很简单，但它是代理学习一切的方式，从下棋到控制机器人，再到优化商业流程。

```{image} /_static/diagrams/AE_loop.png
:width: 50%
:align: center
:class: only-light
```

```{image} /_static/diagrams/AE_loop_dark.png
:width: 50%
:align: center
:class: only-dark
```

## 你的第一个强化学习程序

让我们从一个简单的例子开始，使用 CartPole —— 这是理解基础知识的完美选择：

```python
# 运行 `pip install "gymnasium[classic-control]"` 来安装依赖。
import gymnasium as gym

# 创建我们的训练环境 - 一个需要保持平衡的带杆小车
env = gym.make("CartPole-v1", render_mode="human")

# 重置环境开始新的回合
observation, info = env.reset()
# observation：智能体可以“看到”的信息 - 小车位置、速度、杆角度等
# info：额外的调试信息（通常在基础学习中不需要）

print(f"初始观测值: {observation}")
# 示例输出：[ 0.01234567 -0.00987654  0.02345678  0.01456789]
# [小车位置，小车速度，杆角度，杆角速度]

episode_over = False
total_reward = 0

while not episode_over:
    # 选择一个动作：0 = 推动小车向左，1 = 推动小车向右
    action = env.action_space.sample()  # 目前选择随机动作 - 真实的智能体会更聪明！

    # 执行动作并查看结果
    observation, reward, terminated, truncated, info = env.step(action)

    # reward：每一步杆保持竖直状态就获得+1奖励
    # terminated：如果杆倾斜过度（智能体失败），则为True
    # truncated：如果超过时间限制（500步），则为True

    total_reward += reward
    episode_over = terminated or truncated

print(f"回合结束！总奖励：{total_reward}")
env.close()
```

**你应该看到的内容**：一个窗口会打开，展示一个带杆的小车。小车会随机左右移动，杆子最终会倒下。这是预期的——智能体在随机行动！

### 逐步解释代码

```
.. py:currentmodule:: gymnasium

首先，使用 :meth:`make` 创建一个环境，并可以选择添加一个可选的 `"render_mode"` 参数，指定环境如何可视化。有关不同渲染模式的详细信息，请参见 :meth:`Env.render`。渲染模式决定了你是否能看到可视化窗口（“human”），获得图像数组（“rgb_array”），或者在没有可视化的情况下运行（None - 最适合训练）。

在初始化环境后，我们通过 :meth:`Env.reset` 重置环境，获得第一步的观测值以及额外的信息。这就像是开始一个新游戏或新回合。如果想用特定的随机种子或选项初始化环境（请参阅环境文档获取可能的值），可以使用 `seed` 或 `options` 参数与 :meth:`reset` 一起使用。

由于我们想要在环境结束之前继续智能体-环境循环（结束时间不确定），我们定义了一个变量 `episode_over` 来控制 while 循环。

接下来，智能体在环境中执行一个动作。 :meth:`Env.step` 执行所选的动作（在我们的例子中是通过 `env.action_space.sample()` 随机选择动作），并更新环境。这个动作可以想象成移动一个机器人、按下游戏控制器上的按钮，或做出交易决策。结果，智能体从更新后的环境中获得新的观测值，并获得执行该动作的奖励。对于好的动作（例如成功保持杆子竖直），奖励可能是正的；对于坏的动作（例如让杆子倒下），奖励可能是负的。一个动作-观测值交换被称为 **时间步**。

然而，经过若干时间步后，环境可能会结束——这被称为终止状态。例如，机器人可能已经崩溃，或者成功完成任务，或者我们可能希望在固定的时间步数后停止。在 Gymnasium 中，如果由于任务完成或失败而终止环境， :meth:`step` 会返回 `terminated=True`。如果我们希望在固定的时间步数后结束环境（如时间限制），环境会发出 `truncated=True` 信号。如果 `terminated` 或 `truncated` 为 `True`，我们就结束该回合。在大多数情况下，你会想用 `env.reset()` 重新启动环境，开始一个新的回合。
```

## 动作和观测空间

```
.. py:currentmodule:: gymnasium.Env

每个环境通过 :attr:`action_space` 和 :attr:`observation_space` 属性指定有效动作和观测值的格式。这有助于了解环境的输入和输出，因为所有有效的动作和观测值应该包含在它们各自的空间中。在上述示例中，我们通过 `env.action_space.sample()` 随机选择动作，而不是使用一个智能体策略将观测值映射到动作（这是你将学会构建的内容）。

理解这些空间对于构建智能体至关重要：

- **动作空间**：你的智能体能做什么？（离散选择、连续值等）
- **观测空间**：你的智能体能看到什么？（图像、数字、结构化数据等）

重要的是， :attr:`Env.action_space` 和 :attr:`Env.observation_space` 是 :class:`Space` 的实例，这是一个高级 Python 类，提供了关键功能： :meth:`Space.contains` 和 :meth:`Space.sample`。Gymnasium 支持多种空间类型：

.. py:currentmodule:: gymnasium.spaces

- :class:`Box`：描述有上下限的空间，适用于任何 n 维形状（如连续控制或图像像素）。
- :class:`Discrete`：描述离散空间，其中 `{0, 1, ..., n-1}` 是可能的值（如按钮按压或菜单选择）。
- :class:`MultiBinary`：描述任何 n 维形状的二进制空间（如多个开关）。
- :class:`MultiDiscrete`：由多个 :class:`Discrete` 动作空间组成，每个元素中有不同数量的动作。
- :class:`Text`：描述具有最小和最大长度的字符串空间。
- :class:`Dict`：描述由多个简单空间组成的字典（例如，稍后你将看到的 GridWorld 示例）。
- :class:`Tuple`：描述简单空间的元组。
- :class:`Graph`：描述具有相互连接的节点和边的数学图（网络）。
- :class:`Sequence`：描述由多个简单空间元素组成的可变长度序列。

有关空间的示例用法，请参阅它们的 `文档 </api/spaces>`_ 和 `实用功能 </api/spaces/utils>`_。
```

让我们来看一些示例：

```python
import gymnasium as gym

# 离散动作空间（按钮按压）
env = gym.make("CartPole-v1")
print(f"动作空间: {env.action_space}")  # Discrete(2) - 左或右
print(f"示例动作: {env.action_space.sample()}")  # 0 或 1

# 盒子型观测空间（连续值）
print(f"观测空间: {env.observation_space}")  # Box，包含4个值
# Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
print(f"示例观测: {env.observation_space.sample()}")  # 随机有效观测
```

## 修改环境

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

包装器是一种方便的方式，用于修改现有的环境，而无需直接修改底层代码。可以将包装器视为滤镜或修饰符，用于改变你与环境的交互方式。使用包装器可以避免冗余代码，使环境更加模块化。包装器还可以链式组合，以结合它们的效果。

通过 :meth:`gymnasium.make` 创建的大多数环境默认情况下已经被包装，使用了 :class:`TimeLimit`（在达到最大步数后停止回合）、:class:`OrderEnforcing`（确保正确的重置/步骤顺序）和 :class:`PassiveEnvChecker`（验证你的环境使用情况）。

要包装一个环境，首先初始化一个基础环境，然后将其与可选参数一起传递给包装器的构造函数：
```

```python
>>> import gymnasium as gym
>>> from gymnasium.wrappers import FlattenObservation

>>> # 从一个复杂的观测空间开始
>>> env = gym.make("CarRacing-v3")
>>> env.observation_space.shape
(96, 96, 3)  # 96x96 的 RGB 图像

>>> # 包装环境，将观测空间展平成一个 1D 数组
>>> wrapped_env = FlattenObservation(env)
>>> wrapped_env.observation_space.shape
(27648,)  # 所有像素点都在一个数组中

>>> # 这样可以更方便地与一些期望 1D 输入的算法一起使用
```

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

初学者常用的常见包装器：

- :class:`TimeLimit`：如果超过最大时间步数，发出截断信号（防止无限期的回合）。
- :class:`ClipAction`：将传递给 ``step`` 的任何动作裁剪到有效的动作空间范围内。
- :class:`RescaleAction`：将动作重新缩放到不同的范围（对于输出动作在 [-1, 1] 范围内但环境期望在 [0, 10] 范围内的算法非常有用）。
- :class:`TimeAwareObservation`：向观测中添加当前时间步的信息（有时有助于学习）。
```

翻译如下：

~~~text
有关 Gymnasium 中实现的所有包装器的完整列表，请参见 [wrappers](/api/wrappers)。

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

如果你有一个包装过的环境，并且想要访问所有包装层下的原始环境（例如手动调用某个函数或更改某些底层方面），可以使用 :attr:`unwrapped` 属性。如果环境已经是基本环境，:attr:`unwrapped` 将直接返回它本身。
>>> wrapped_env
<FlattenObservation<TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v3>>>>>>
>>> wrapped_env.unwrapped
<gymnasium.envs.box2d.car_racing.CarRacing object at 0x7f04efcb8850>
~~~

## 初学者常见问题

**智能体行为：**

- 智能体随机执行：当使用 `env.action_space.sample()` 时，这是预期的行为！真正的学习发生在你用智能策略替换它时。
- 回合立即结束：检查是否在回合之间正确处理了重置。

**常见代码错误：**

```python
# ❌ Wrong - forgetting to reset
env = gym.make("CartPole-v1")
obs, reward, terminated, truncated, info = env.step(action)  # Error!

# ✅ Correct - always reset first
env = gym.make("CartPole-v1")
obs, info = env.reset()  # Start properly
obs, reward, terminated, truncated, info = env.step(action)  # Now this works
```

## 下一步

现在你已经理解了基础知识，接下来可以：

1. **[训练一个实际的智能体](https://chatgpt.com/c/train_agent)** - 用智能替代随机动作
2. **[创建自定义环境](https://chatgpt.com/c/create_custom_env)** - 构建你自己的强化学习问题
3. **[记录智能体行为](https://chatgpt.com/c/record_agent)** - 保存训练中的视频和数据
4. **[加速训练](https://chatgpt.com/c/speed_up_env)** - 使用向量化环境和其他优化方法








---

# Basic Usage

## What is Reinforcement Learning?

Before diving into Gymnasium, let's understand what we're trying to achieve. Reinforcement learning is like teaching through trial and error - an agent learns by trying actions, receiving feedback (rewards), and gradually improving its behavior. Think of training a pet with treats, learning to ride a bike through practice, or mastering a video game by playing it repeatedly.

The key insight is that we don't tell the agent exactly what to do. Instead, we create an environment where it can experiment safely and learn from the consequences of its actions.

## Why Gymnasium?

```{eval-rst}
.. py:currentmodule:: gymnasium

Whether you want to train an agent to play games, control robots, or optimize trading strategies, Gymnasium gives you the tools to build and test your ideas.
At its heart, Gymnasium provides an API (application programming interface) for all single agent reinforcement learning environments, with implementations of common environments: cartpole, pendulum, mountain-car, mujoco, atari, and more. This page will outline the basics of how to use Gymnasium including its four key functions: :meth:`make`, :meth:`Env.reset`, :meth:`Env.step` and :meth:`Env.render`.

At the core of Gymnasium is :class:`Env`, a high-level python class representing a markov decision process (MDP) from reinforcement learning theory (note: this is not a perfect reconstruction, missing several components of MDPs). The class provides users the ability to start new episodes, take actions and visualize the agent's current state. Alongside :class:`Env`, :class:`Wrapper` are provided to help augment / modify the environment, in particular, the agent observations, rewards and actions taken.
```

## Initializing Environments

```{eval-rst}
.. py:currentmodule:: gymnasium

Initializing environments is very easy in Gymnasium and can be done via the :meth:`make` function:
```

```python
import gymnasium as gym

# Create a simple environment perfect for beginners
env = gym.make('CartPole-v1')

# The CartPole environment: balance a pole on a moving cart
# - Simple but not trivial
# - Fast training
# - Clear success/failure criteria
```

```{eval-rst}
.. py:currentmodule:: gymnasium

This function will return an :class:`Env` for users to interact with. To see all environments you can create, use :meth:`pprint_registry`. Furthermore, :meth:`make` provides a number of additional arguments for specifying keywords to the environment, adding more or less wrappers, etc. See :meth:`make` for more information.
```

## Understanding the Agent-Environment Loop

In reinforcement learning, the classic "agent-environment loop" pictured below represents how learning happens in RL. It's simpler than it might first appear:

1. **Agent observes** the current situation (like looking at a game screen)
2. **Agent chooses an action** based on what it sees (like pressing a button)
3. **Environment responds** with a new situation and a reward (game state changes, score updates)
4. **Repeat** until the episode ends

This might seem simple, but it's how agents learn everything from playing chess to controlling robots to optimizing business processes.

```{image} /_static/diagrams/AE_loop.png
:width: 50%
:align: center
:class: only-light
```

```{image} /_static/diagrams/AE_loop_dark.png
:width: 50%
:align: center
:class: only-dark
```

## Your First RL Program

Let's start with a simple example using CartPole - perfect for understanding the basics:

```python
# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1", render_mode="human")

# Reset environment to start a new episode
observation, info = env.reset()
# observation: what the agent can "see" - cart position, velocity, pole angle, etc.
# info: extra debugging information (usually not needed for basic learning)

print(f"Starting observation: {observation}")
# Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action: 0 = push cart left, 1 = push cart right
    action = env.action_space.sample()  # Random action for now - real agents will be smarter!

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
```

**What you should see**: A window opens showing a cart with a pole. The cart moves randomly left and right, and the pole eventually falls over. This is expected - the agent is acting randomly!

### Explaining the Code Step by Step

```{eval-rst}
.. py:currentmodule:: gymnasium

First, an environment is created using :meth:`make` with an optional ``"render_mode"`` parameter that specifies how the environment should be visualized. See :meth:`Env.render` for details on different render modes. The render mode determines whether you see a visual window ("human"), get image arrays ("rgb_array"), or run without visuals (None - fastest for training).

After initializing the environment, we :meth:`Env.reset` the environment to get the first observation along with additional information. This is like starting a new game or episode. For initializing the environment with a particular random seed or options (see the environment documentation for possible values) use the ``seed`` or ``options`` parameters with :meth:`reset`.

As we want to continue the agent-environment loop until the environment ends (which happens in an unknown number of timesteps), we define ``episode_over`` as a variable to control our while loop.

Next, the agent performs an action in the environment. :meth:`Env.step` executes the selected action (in our example, random with ``env.action_space.sample()``) to update the environment. This action can be imagined as moving a robot, pressing a button on a game controller, or making a trading decision. As a result, the agent receives a new observation from the updated environment along with a reward for taking the action. This reward could be positive for good actions (like successfully balancing the pole) or negative for bad actions (like letting the pole fall). One such action-observation exchange is called a **timestep**.

However, after some timesteps, the environment may end - this is called the terminal state. For instance, the robot may have crashed, or succeeded in completing a task, or we may want to stop after a fixed number of timesteps. In Gymnasium, if the environment has terminated due to the task being completed or failed, this is returned by :meth:`step` as ``terminated=True``. If we want the environment to end after a fixed number of timesteps (like a time limit), the environment issues a ``truncated=True`` signal. If either ``terminated`` or ``truncated`` are ``True``, we end the episode. In most cases, you'll want to restart the environment with ``env.reset()`` to begin a new episode.
```

## Action and observation spaces

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

Every environment specifies the format of valid actions and observations with the :attr:`action_space` and :attr:`observation_space` attributes. This is helpful for knowing both the expected input and output of the environment, as all valid actions and observations should be contained within their respective spaces. In the example above, we sampled random actions via ``env.action_space.sample()`` instead of using an intelligent agent policy that maps observations to actions (which is what you'll learn to build).

Understanding these spaces is crucial for building agents:
- **Action Space**: What can your agent do? (discrete choices, continuous values, etc.)
- **Observation Space**: What can your agent see? (images, numbers, structured data, etc.)

Importantly, :attr:`Env.action_space` and :attr:`Env.observation_space` are instances of :class:`Space`, a high-level python class that provides key functions: :meth:`Space.contains` and :meth:`Space.sample`. Gymnasium supports a wide range of spaces:

.. py:currentmodule:: gymnasium.spaces

- :class:`Box`: describes bounded space with upper and lower limits of any n-dimensional shape (like continuous control or image pixels).
- :class:`Discrete`: describes a discrete space where ``{0, 1, ..., n-1}`` are the possible values (like button presses or menu choices).
- :class:`MultiBinary`: describes a binary space of any n-dimensional shape (like multiple on/off switches).
- :class:`MultiDiscrete`: consists of a series of :class:`Discrete` action spaces with different numbers of actions in each element.
- :class:`Text`: describes a string space with minimum and maximum length.
- :class:`Dict`: describes a dictionary of simpler spaces (like our GridWorld example you'll see later).
- :class:`Tuple`: describes a tuple of simple spaces.
- :class:`Graph`: describes a mathematical graph (network) with interlinking nodes and edges.
- :class:`Sequence`: describes a variable length of simpler space elements.

For example usage of spaces, see their `documentation </api/spaces>`_ along with `utility functions </api/spaces/utils>`_.
```

Let's look at some examples:

```python
import gymnasium as gym

# Discrete action space (button presses)
env = gym.make("CartPole-v1")
print(f"Action space: {env.action_space}")  # Discrete(2) - left or right
print(f"Sample action: {env.action_space.sample()}")  # 0 or 1

# Box observation space (continuous values)
print(f"Observation space: {env.observation_space}")  # Box with 4 values
# Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
print(f"Sample observation: {env.observation_space.sample()}")  # Random valid observation
```

## Modifying the environment

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

Wrappers are a convenient way to modify an existing environment without having to alter the underlying code directly. Think of wrappers like filters or modifiers that change how you interact with an environment. Using wrappers allows you to avoid boilerplate code and make your environment more modular. Wrappers can also be chained to combine their effects.

Most environments created via :meth:`gymnasium.make` will already be wrapped by default using :class:`TimeLimit` (stops episodes after a maximum number of steps), :class:`OrderEnforcing` (ensures proper reset/step order), and :class:`PassiveEnvChecker` (validates your environment usage).

To wrap an environment, you first initialize a base environment, then pass it along with optional parameters to the wrapper's constructor:
```

```python
>>> import gymnasium as gym
>>> from gymnasium.wrappers import FlattenObservation

>>> # Start with a complex observation space
>>> env = gym.make("CarRacing-v3")
>>> env.observation_space.shape
(96, 96, 3)  # 96x96 RGB image

>>> # Wrap it to flatten the observation into a 1D array
>>> wrapped_env = FlattenObservation(env)
>>> wrapped_env.observation_space.shape
(27648,)  # All pixels in a single array

>>> # This makes it easier to use with some algorithms that expect 1D input
```

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

Common wrappers that beginners find useful:

- :class:`TimeLimit`: Issues a truncated signal if a maximum number of timesteps has been exceeded (preventing infinite episodes).
- :class:`ClipAction`: Clips any action passed to ``step`` to ensure it's within the valid action space.
- :class:`RescaleAction`: Rescales actions to a different range (useful for algorithms that output actions in [-1, 1] but environment expects [0, 10]).
- :class:`TimeAwareObservation`: Adds information about the current timestep to the observation (sometimes helps with learning).
```

For a full list of implemented wrappers in Gymnasium, see [wrappers](/api/wrappers).

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

If you have a wrapped environment and want to access the original environment underneath all the layers of wrappers (to manually call a function or change some underlying aspect), you can use the :attr:`unwrapped` attribute. If the environment is already a base environment, :attr:`unwrapped` just returns itself.
```

```python
>>> wrapped_env
<FlattenObservation<TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v3>>>>>>
>>> wrapped_env.unwrapped
<gymnasium.envs.box2d.car_racing.CarRacing object at 0x7f04efcb8850>
```

## Common Issues for Beginners

**Agent Behavior:**

- Agent performs randomly: That's expected when using `env.action_space.sample()`! Real learning happens when you replace this with an intelligent policy
- Episodes end immediately: Check if you're properly handling the reset between episodes

**Common Code Mistakes:**

```python
# ❌ Wrong - forgetting to reset
env = gym.make("CartPole-v1")
obs, reward, terminated, truncated, info = env.step(action)  # Error!

# ✅ Correct - always reset first
env = gym.make("CartPole-v1")
obs, info = env.reset()  # Start properly
obs, reward, terminated, truncated, info = env.step(action)  # Now this works
```

## Next Steps

Now that you understand the basics, you're ready to:

1. **[Train an actual agent](train_agent)** - Replace random actions with intelligence
2. **[Create custom environments](create_custom_env)** - Build your own RL problems
3. **[Record agent behavior](record_agent)** - Save videos and data from training
4. **[Speed up training](speed_up_env)** - Use vectorized environments and other optimizations
