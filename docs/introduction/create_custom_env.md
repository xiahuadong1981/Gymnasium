---
布局： "内容"
标题： 创建自定义环境
---

# 创建自定义环境

## 编码前：环境设计

创建一个强化学习（RL）环境就像设计一个视频游戏或模拟系统。在编写任何代码之前，你需要思考你想要解决的学习问题。这个设计阶段至关重要——一个设计不好的环境将使学习变得困难或不可能，无论你的算法有多优秀。

### 关键设计问题

问问自己以下这些基本问题：

**🎯 代理需要学习什么技能？**

* 穿越迷宫？
* 平衡和控制一个系统？
* 优化资源分配？
* 玩一个战略游戏？

**👀 代理需要哪些信息？**

* 位置和速度？
* 系统的当前状态？
* 历史数据？
* 部分或完全可观察？

**🎮 代理可以采取哪些行动？**

* 离散选择（上/下/左/右移动）？
* 连续控制（转向角度、油门）？
* 多个同时的动作？

**🏆 如何衡量成功？**

* 达到特定目标？
* 最小化时间或能量？
* 最大化得分？
* 避免失败？

**⏰ 什么时候应该结束每一回合？**

* 任务完成（成功/失败）？
* 时间限制？
* 安全约束？

### GridWorld 示例设计

在我们的教程示例中，我们将创建一个简单的 GridWorld 环境：

* **🎯 技能**：高效地导航到目标位置
* **👀 信息**：代理的位置和网格上的目标位置
* **🎮 行动**：向上、向下、向左或向右移动
* **🏆 成功**：在最少的步数内到达目标
* **⏰ 结束**：当代理到达目标时（或可选的时间限制）

这提供了一个清晰的学习问题，足够简单易懂，但要优化解决并不简单。

---

本页面提供了使用 Gymnasium 创建自定义环境的完整实现。欲了解更完整的[教程](../tutorials/gymnasium_basics/environment_creation)（带渲染）。

我们建议在阅读本页面之前，先熟悉[基础用法](basic_usage)！

我们将实现我们的 GridWorld 游戏为一个固定大小的二维方格。代理可以在每个时间步长中在网格单元之间垂直或水平移动，目标是导航到一个在回合开始时随机放置的目标。

## 环境 `__init__`


```{eval-rst}
.. py:currentmodule:: gymnasium

与所有环境一样，我们的自定义环境将继承自 :class:`gymnasium.Env`，该类定义了所有环境必须遵循的结构。一个要求是定义观测空间和动作空间，声明哪些输入（动作）和输出（观测）在这个环境中是有效的。

正如我们设计中所概述的，我们的代理有四个离散的动作（在四个主要方向上移动），所以我们将使用 `Discrete(4)` 空间。

```

```{eval-rst}
.. py:currentmodule:: gymnasium.spaces

对于我们的观测空间，我们有几种选择。我们可以将整个网格表示为一个二维数组，或者使用坐标位置，甚至使用一个包含单独“层”的三维数组来表示代理和目标。对于本教程，我们将使用一个简单的字典格式，例如 `{"agent": array([1, 0]), "target": array([0, 3])}`，其中数组表示 x,y 坐标。

这种选择使得观测空间更加易于人类阅读并且便于调试。我们将其声明为一个 :class:`Dict` 空间，其中代理和目标空间是 :class:`Box` 空间，包含整数坐标。

```

有关可以与环境一起使用的所有空间的完整列表，请参见 [spaces](../api/spaces)。

```python
from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid (5x5 by default)
        self.size = size

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
        }
```

## 构建观测空间

```{eval-rst}
.. py:currentmodule:: gymnasium

由于我们需要在 :meth:`Env.reset` 和 :meth:`Env.step` 中计算观测空间，因此拥有一个辅助方法 `_get_obs` 来将环境的内部状态转换为观测格式是很方便的。这可以保持我们的代码遵循 DRY（不要重复自己）原则，并且使以后修改观测格式更加容易。

```

```python
    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self._agent_location, "target": self._target_location}
```

```{eval-rst}
.. py:currentmodule:: gymnasium

我们还可以为 :meth:`Env.reset` 和 :meth:`Env.step` 返回的辅助信息实现一个类似的方法。在我们的例子中，我们将提供代理和目标之间的曼哈顿距离——这对于调试和理解代理的进展非常有用，但不应被学习算法本身使用。

```

```python
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
```

```{eval-rst}
.. py:currentmodule:: gymnasium

有时，info 会包含仅在 :meth:`Env.step` 内部可用的数据（如单独的奖励组成部分、动作成功/失败等）。在这种情况下，我们会直接在 step 方法中更新 `_get_info` 返回的字典。
```

## Reset function

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

:meth:`reset` 方法开始一个新的回合。它有两个可选参数：`seed` 用于可重复的随机生成，`options` 用于额外的配置。在第一行，你必须调用 `super().reset(seed=seed)` 来正确初始化随机数生成器。

在我们的 GridWorld 环境中，:meth:`reset` 会随机将代理和目标放置在网格上，并确保它们不会在同一位置开始。我们将初始观测空间和信息作为元组返回。

```

```python
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
```

## Step function

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

:meth:`step` 方法包含核心的环境逻辑。它接受一个动作，更新环境状态，并返回结果。这里包含了物理学、游戏规则和奖励逻辑。

对于 GridWorld，我们需要：

1. 将离散动作转换为移动方向
2. 更新代理的位置（并进行边界检查）
3. 根据是否到达目标来计算奖励
4. 确定回合是否结束
5. 返回所有必要的信息
```

```python
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
```

## Common Environment Design Pitfalls

Now that you've seen the basic structure, let's discuss common mistakes beginners make:

### Reward Design Issues

**Problem**: Only rewarding at the very end (sparse rewards)
```python
# This makes learning very difficult!
reward = 1 if terminated else 0
```

**Better**: Provide intermediate feedback
```python
# Option 1: Small step penalty to encourage efficiency
reward = 1 if terminated else -0.01

# Option 2: Distance-based reward shaping
distance = np.linalg.norm(self._agent_location - self._target_location)
reward = 1 if terminated else -0.1 * distance
```

### State Representation Problems

**Problem**: Including irrelevant information or missing crucial details
```python
# Too much info - agent doesn't need grid size in every observation
obs = {"agent": self._agent_location, "target": self._target_location, "size": self.size}

# Too little info - agent can't distinguish different positions
obs = {"distance": distance}  # Missing actual positions!
```

**Better**: Include exactly what's needed for optimal decisions
```python
# Just right - positions are sufficient for navigation
obs = {"agent": self._agent_location, "target": self._target_location}
```

### Action Space Issues

**Problem**: Actions that don't make sense or are impossible to execute
```python
# Bad: Agent can move diagonally but environment doesn't support it
self.action_space = gym.spaces.Discrete(8)  # 8 directions including diagonals

# Bad: Continuous actions for discrete movement
self.action_space = gym.spaces.Box(-1, 1, shape=(2,))  # Continuous x,y movement
```

### Boundary Handling Errors

**Problem**: Allowing invalid states or unclear boundary behavior
```python
# Bad: Agent can go outside the grid
self._agent_location = self._agent_location + direction  # No bounds checking!

# Unclear: What happens when agent hits wall?
if np.any(self._agent_location < 0) or np.any(self._agent_location >= self.size):
    # Do nothing? Reset episode? Give penalty? Unclear!
```

**Better**: Clear, consistent boundary handling
```python
# Clear: Agent stays in place when hitting boundaries
self._agent_location = np.clip(
    self._agent_location + direction, 0, self.size - 1
)
```

## Registering and making the environment

```{eval-rst}
While you can use your custom environment immediately, it's more convenient to register it with Gymnasium so you can create it with :meth:`gymnasium.make` just like built-in environments.

The environment ID has three components: an optional namespace (here: ``gymnasium_env``), a mandatory name (here: ``GridWorld``), and an optional but recommended version (here: v0). You could register it as ``GridWorld-v0``, ``GridWorld``, or ``gymnasium_env/GridWorld``, but the full format is recommended for clarity.

Since this tutorial isn't part of a Python package, we pass the class directly as the entry point. In real projects, you'd typically use a string like ``"my_package.envs:GridWorldEnv"``.
```

```python
# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)
```

For a more complete guide on registering custom environments (including with string entry points), please read the full [create environment](../tutorials/gymnasium_basics/environment_creation) tutorial.

```{eval-rst}
Once registered, you can check all available environments with :meth:`gymnasium.pprint_registry` and create instances with :meth:`gymnasium.make`. You can also create vectorized versions with :meth:`gymnasium.make_vec`.
```

```python
import gymnasium as gym

# Create the environment like any built-in environment
>>> env = gym.make("gymnasium_env/GridWorld-v0")
<OrderEnforcing<PassiveEnvChecker<GridWorld<gymnasium_env/GridWorld-v0>>>>

# Customize environment parameters
>>> env = gym.make("gymnasium_env/GridWorld-v0", size=10)
>>> env.unwrapped.size
10

# Create multiple environments for parallel training
>>> vec_env = gym.make_vec("gymnasium_env/GridWorld-v0", num_envs=3)
SyncVectorEnv(gymnasium_env/GridWorld-v0, num_envs=3)
```


## Debugging Your Environment

When your environment doesn't work as expected, here are common debugging strategies:

### Check Environment Validity
```python
from gymnasium.utils.env_checker import check_env

# This will catch many common issues
try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
```

### Manual Testing with Known Actions
```python
# Test specific action sequences to verify behavior
env = gym.make("gymnasium_env/GridWorld-v0")
obs, info = env.reset(seed=42)  # Use seed for reproducible testing

print(f"Starting position - Agent: {obs['agent']}, Target: {obs['target']}")

# Test each action type
actions = [0, 1, 2, 3]  # right, up, left, down
for action in actions:
    old_pos = obs['agent'].copy()
    obs, reward, terminated, truncated, info = env.step(action)
    new_pos = obs['agent']
    print(f"Action {action}: {old_pos} -> {new_pos}, reward={reward}")
```

### Common Debug Issues
```python
# Issue 1: Forgot to call super().reset()
def reset(self, seed=None, options=None):
    # super().reset(seed=seed)  # ❌ Missing this line
    # Results in: possibly incorrect seeding

# Issue 2: Mixing up coordinate conventions
# Using Cartesian [x, y] instead of NumPy [row, col] causes visual confusion
# when rendering, since row 0 is at the top of the screen:
self._action_to_direction = {
    0: np.array([1, 0]),   # Intended as "right" but this changes row
    1: np.array([0, 1]),   # Intended as "up" but this changes column
}

# Issue 3: Not handling boundaries properly
# This allows agent to go outside the grid!
self._agent_location = self._agent_location + direction  # ❌ No bounds checking
```

## Using Wrappers

Sometimes you want to modify your environment's behavior without changing the core implementation. Wrappers are perfect for this - they let you add functionality like changing observation formats, adding time limits, or modifying rewards without touching your original environment code.

```python
>>> from gymnasium.wrappers import FlattenObservation

>>> # Original observation is a dictionary
>>> env = gym.make('gymnasium_env/GridWorld-v0')
>>> env.observation_space
Dict('agent': Box(0, 4, (2,), int64), 'target': Box(0, 4, (2,), int64))

>>> obs, info = env.reset()
>>> obs
{'agent': array([4, 1]), 'target': array([2, 4])}

>>> # Wrap it to flatten observations into a single array
>>> wrapped_env = FlattenObservation(env)
>>> wrapped_env.observation_space
Box(0, 4, (4,), int64)

>>> obs, info = wrapped_env.reset()
>>> obs
array([3, 0, 2, 1])  # [agent_x, agent_y, target_x, target_y]
```

This is particularly useful when working with algorithms that expect specific input formats (like neural networks that need 1D arrays instead of dictionaries).

## Advanced Environment Features

Once you have the basics working, you might want to add more sophisticated features:

### Adding Rendering
```python
def render(self):
    """Render the environment for human viewing."""
    if self.render_mode == "human":
        # Print a simple ASCII representation
        for y in range(self.size - 1, -1, -1):  # Top to bottom
            row = ""
            for x in range(self.size):
                if np.array_equal([x, y], self._agent_location):
                    row += "A "  # Agent
                elif np.array_equal([x, y], self._target_location):
                    row += "T "  # Target
                else:
                    row += ". "  # Empty
            print(row)
        print()
```

### Parameterized Environments
```python
def __init__(self, size: int = 5, reward_scale: float = 1.0, step_penalty: float = 0.0):
    self.size = size
    self.reward_scale = reward_scale
    self.step_penalty = step_penalty
    # ... rest of init ...

def step(self, action):
    # ... movement logic ...

    # Flexible reward calculation
    if terminated:
        reward = self.reward_scale  # Success reward
    else:
        reward = -self.step_penalty  # Step penalty (0 by default)
```

## Real-World Environment Design Tips

### Start Simple, Add Complexity Gradually
1. **First**: Get basic movement and goal-reaching working
2. **Then**: Add obstacles, multiple goals, or time pressure
3. **Finally**: Add complex dynamics, partial observability, or multi-agent interactions

### Design for Learning
- **Clear Success Criteria**: Agent should know when it's doing well
- **Reasonable Difficulty**: Not too easy (trivial) or too hard (impossible)
- **Consistent Rules**: Same action in same state should have same effect
- **Informative Observations**: Include everything needed for optimal decisions

### Think About Your Research Question
- **Navigation**: Focus on spatial reasoning and path planning
- **Control**: Emphasize dynamics, stability, and continuous actions
- **Strategy**: Include partial information, opponent modeling, or long-term planning
- **Optimization**: Design clear trade-offs and resource constraints

## Next Steps

Congratulations! You now know how to create custom RL environments. Here's what to explore next:

1. **Add rendering** to visualize your environment ([complete tutorial](../tutorials/gymnasium_basics/environment_creation))
2. **Train an agent** on your custom environment ([training guide](train_agent))
3. **Experiment with different reward functions** to see how they affect learning
4. **Try wrapper combinations** to modify your environment's behavior
5. **Create more complex environments** with obstacles, multiple agents, or continuous actions

The key to good environment design is iteration - start simple, test thoroughly, and gradually add complexity as needed for your research or application goals.






---
layout: "contents"
title: Create custom env
---

# Create a Custom Environment

## Before You Code: Environment Design

Creating an RL environment is like designing a video game or simulation. Before writing any code, you need to think through the learning problem you want to solve. This design phase is crucial - a poorly designed environment will make learning difficult or impossible, no matter how good your algorithm is.

### Key Design Questions

Ask yourself these fundamental questions:

**🎯 What skill should the agent learn?**
- Navigate through a maze?
- Balance and control a system?
- Optimize resource allocation?
- Play a strategic game?

**👀 What information does the agent need?**
- Position and velocity?
- Current state of the system?
- Historical data?
- Partial or full observability?

**🎮 What actions can the agent take?**
- Discrete choices (move up/down/left/right)?
- Continuous control (steering angle, throttle)?
- Multiple simultaneous actions?

**🏆 How do we measure success?**
- Reaching a specific goal?
- Minimizing time or energy?
- Maximizing a score?
- Avoiding failures?

**⏰ When should episodes end?**
- Task completion (success/failure)?
- Time limits?
- Safety constraints?

### GridWorld Example Design

For our tutorial example, we'll create a simple GridWorld environment:

- **🎯 Skill**: Navigate efficiently to a target location
- **👀 Information**: Agent position and target position on a grid
- **🎮 Actions**: Move up, down, left, or right
- **🏆 Success**: Reach the target in minimum steps
- **⏰ End**: When agent reaches target (or optional time limit)

This provides a clear learning problem that's simple enough to understand but non-trivial to solve optimally.

---

This page provides a complete implementation of creating custom environments with Gymnasium. For a more [complete tutorial](../tutorials/gymnasium_basics/environment_creation) with rendering.

We recommend that you familiarise yourself with the [basic usage](basic_usage) before reading this page!

We will implement our GridWorld game as a 2-dimensional square grid of fixed size. The agent can move vertically or horizontally between grid cells in each timestep, and the goal is to navigate to a target that has been placed randomly at the beginning of the episode.

## Environment `__init__`

```{eval-rst}
.. py:currentmodule:: gymnasium

Like all environments, our custom environment will inherit from :class:`gymnasium.Env` that defines the structure all environments must follow. One of the requirements is defining the observation and action spaces, which declare what inputs (actions) and outputs (observations) are valid for this environment.

As outlined in our design, our agent has four discrete actions (move in cardinal directions), so we'll use ``Discrete(4)`` space.
```

```{eval-rst}
.. py:currentmodule:: gymnasium.spaces

For our observation, we have several options. We could represent the full grid as a 2D array, or use coordinate positions, or even a 3D array with separate "layers" for agent and target. For this tutorial, we'll use a simple dictionary format like ``{"agent": array([1, 0]), "target": array([0, 3])}`` where the arrays represent x,y coordinates.

This choice makes the observation human-readable and easy to debug. We'll declare this as a :class:`Dict` space with the agent and target spaces being :class:`Box` spaces that contain integer coordinates.
```

For a full list of possible spaces to use with an environment, see [spaces](../api/spaces)

```python
from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid (5x5 by default)
        self.size = size

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
        }
```

## Constructing Observations

```{eval-rst}
.. py:currentmodule:: gymnasium

Since we need to compute observations in both :meth:`Env.reset` and :meth:`Env.step`, it's convenient to have a helper method ``_get_obs`` that translates the environment's internal state into the observation format. This keeps our code DRY (Don't Repeat Yourself) and makes it easier to modify the observation format later.
```

```python
    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self._agent_location, "target": self._target_location}
```

```{eval-rst}
.. py:currentmodule:: gymnasium

We can also implement a similar method for auxiliary information returned by :meth:`Env.reset` and :meth:`Env.step`. In our case, we'll provide the Manhattan distance between agent and target - this can be useful for debugging and understanding agent progress, but shouldn't be used by the learning algorithm itself.
```

```python
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
```

```{eval-rst}
.. py:currentmodule:: gymnasium

Sometimes info will contain data that's only available inside :meth:`Env.step` (like individual reward components, action success/failure, etc.). In those cases, we'd update the dictionary returned by ``_get_info`` directly in the step method.
```

## Reset function

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

The :meth:`reset` method starts a new episode. It takes two optional parameters: ``seed`` for reproducible random generation and ``options`` for additional configuration. On the first line, you must call ``super().reset(seed=seed)`` to properly initialize the random number generator.

In our GridWorld environment, :meth:`reset` randomly places the agent and target on the grid, ensuring they don't start in the same location. We return both the initial observation and info as a tuple.
```

```python
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
```

## Step function

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

The :meth:`step` method contains the core environment logic. It takes an action, updates the environment state, and returns the results. This is where the physics, game rules, and reward logic live.

For GridWorld, we need to:
1. Convert the discrete action to a movement direction
2. Update the agent's position (with boundary checking)
3. Calculate the reward based on whether the target was reached
4. Determine if the episode should end
5. Return all the required information
```

```python
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
```

## Common Environment Design Pitfalls

Now that you've seen the basic structure, let's discuss common mistakes beginners make:

### Reward Design Issues

**Problem**: Only rewarding at the very end (sparse rewards)
```python
# This makes learning very difficult!
reward = 1 if terminated else 0
```

**Better**: Provide intermediate feedback
```python
# Option 1: Small step penalty to encourage efficiency
reward = 1 if terminated else -0.01

# Option 2: Distance-based reward shaping
distance = np.linalg.norm(self._agent_location - self._target_location)
reward = 1 if terminated else -0.1 * distance
```

### State Representation Problems

**Problem**: Including irrelevant information or missing crucial details
```python
# Too much info - agent doesn't need grid size in every observation
obs = {"agent": self._agent_location, "target": self._target_location, "size": self.size}

# Too little info - agent can't distinguish different positions
obs = {"distance": distance}  # Missing actual positions!
```

**Better**: Include exactly what's needed for optimal decisions
```python
# Just right - positions are sufficient for navigation
obs = {"agent": self._agent_location, "target": self._target_location}
```

### Action Space Issues

**Problem**: Actions that don't make sense or are impossible to execute
```python
# Bad: Agent can move diagonally but environment doesn't support it
self.action_space = gym.spaces.Discrete(8)  # 8 directions including diagonals

# Bad: Continuous actions for discrete movement
self.action_space = gym.spaces.Box(-1, 1, shape=(2,))  # Continuous x,y movement
```

### Boundary Handling Errors

**Problem**: Allowing invalid states or unclear boundary behavior
```python
# Bad: Agent can go outside the grid
self._agent_location = self._agent_location + direction  # No bounds checking!

# Unclear: What happens when agent hits wall?
if np.any(self._agent_location < 0) or np.any(self._agent_location >= self.size):
    # Do nothing? Reset episode? Give penalty? Unclear!
```

**Better**: Clear, consistent boundary handling
```python
# Clear: Agent stays in place when hitting boundaries
self._agent_location = np.clip(
    self._agent_location + direction, 0, self.size - 1
)
```

## Registering and making the environment

```{eval-rst}
While you can use your custom environment immediately, it's more convenient to register it with Gymnasium so you can create it with :meth:`gymnasium.make` just like built-in environments.

The environment ID has three components: an optional namespace (here: ``gymnasium_env``), a mandatory name (here: ``GridWorld``), and an optional but recommended version (here: v0). You could register it as ``GridWorld-v0``, ``GridWorld``, or ``gymnasium_env/GridWorld``, but the full format is recommended for clarity.

Since this tutorial isn't part of a Python package, we pass the class directly as the entry point. In real projects, you'd typically use a string like ``"my_package.envs:GridWorldEnv"``.
```

```python
# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)
```

For a more complete guide on registering custom environments (including with string entry points), please read the full [create environment](../tutorials/gymnasium_basics/environment_creation) tutorial.

```{eval-rst}
Once registered, you can check all available environments with :meth:`gymnasium.pprint_registry` and create instances with :meth:`gymnasium.make`. You can also create vectorized versions with :meth:`gymnasium.make_vec`.
```

```python
import gymnasium as gym

# Create the environment like any built-in environment
>>> env = gym.make("gymnasium_env/GridWorld-v0")
<OrderEnforcing<PassiveEnvChecker<GridWorld<gymnasium_env/GridWorld-v0>>>>

# Customize environment parameters
>>> env = gym.make("gymnasium_env/GridWorld-v0", size=10)
>>> env.unwrapped.size
10

# Create multiple environments for parallel training
>>> vec_env = gym.make_vec("gymnasium_env/GridWorld-v0", num_envs=3)
SyncVectorEnv(gymnasium_env/GridWorld-v0, num_envs=3)
```


## Debugging Your Environment

When your environment doesn't work as expected, here are common debugging strategies:

### Check Environment Validity
```python
from gymnasium.utils.env_checker import check_env

# This will catch many common issues
try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
```

### Manual Testing with Known Actions
```python
# Test specific action sequences to verify behavior
env = gym.make("gymnasium_env/GridWorld-v0")
obs, info = env.reset(seed=42)  # Use seed for reproducible testing

print(f"Starting position - Agent: {obs['agent']}, Target: {obs['target']}")

# Test each action type
actions = [0, 1, 2, 3]  # right, up, left, down
for action in actions:
    old_pos = obs['agent'].copy()
    obs, reward, terminated, truncated, info = env.step(action)
    new_pos = obs['agent']
    print(f"Action {action}: {old_pos} -> {new_pos}, reward={reward}")
```

### Common Debug Issues
```python
# Issue 1: Forgot to call super().reset()
def reset(self, seed=None, options=None):
    # super().reset(seed=seed)  # ❌ Missing this line
    # Results in: possibly incorrect seeding

# Issue 2: Mixing up coordinate conventions
# Using Cartesian [x, y] instead of NumPy [row, col] causes visual confusion
# when rendering, since row 0 is at the top of the screen:
self._action_to_direction = {
    0: np.array([1, 0]),   # Intended as "right" but this changes row
    1: np.array([0, 1]),   # Intended as "up" but this changes column
}

# Issue 3: Not handling boundaries properly
# This allows agent to go outside the grid!
self._agent_location = self._agent_location + direction  # ❌ No bounds checking
```

## Using Wrappers

Sometimes you want to modify your environment's behavior without changing the core implementation. Wrappers are perfect for this - they let you add functionality like changing observation formats, adding time limits, or modifying rewards without touching your original environment code.

```python
>>> from gymnasium.wrappers import FlattenObservation

>>> # Original observation is a dictionary
>>> env = gym.make('gymnasium_env/GridWorld-v0')
>>> env.observation_space
Dict('agent': Box(0, 4, (2,), int64), 'target': Box(0, 4, (2,), int64))

>>> obs, info = env.reset()
>>> obs
{'agent': array([4, 1]), 'target': array([2, 4])}

>>> # Wrap it to flatten observations into a single array
>>> wrapped_env = FlattenObservation(env)
>>> wrapped_env.observation_space
Box(0, 4, (4,), int64)

>>> obs, info = wrapped_env.reset()
>>> obs
array([3, 0, 2, 1])  # [agent_x, agent_y, target_x, target_y]
```

This is particularly useful when working with algorithms that expect specific input formats (like neural networks that need 1D arrays instead of dictionaries).

## Advanced Environment Features

Once you have the basics working, you might want to add more sophisticated features:

### Adding Rendering
```python
def render(self):
    """Render the environment for human viewing."""
    if self.render_mode == "human":
        # Print a simple ASCII representation
        for y in range(self.size - 1, -1, -1):  # Top to bottom
            row = ""
            for x in range(self.size):
                if np.array_equal([x, y], self._agent_location):
                    row += "A "  # Agent
                elif np.array_equal([x, y], self._target_location):
                    row += "T "  # Target
                else:
                    row += ". "  # Empty
            print(row)
        print()
```

### Parameterized Environments
```python
def __init__(self, size: int = 5, reward_scale: float = 1.0, step_penalty: float = 0.0):
    self.size = size
    self.reward_scale = reward_scale
    self.step_penalty = step_penalty
    # ... rest of init ...

def step(self, action):
    # ... movement logic ...

    # Flexible reward calculation
    if terminated:
        reward = self.reward_scale  # Success reward
    else:
        reward = -self.step_penalty  # Step penalty (0 by default)
```

## Real-World Environment Design Tips

### Start Simple, Add Complexity Gradually
1. **First**: Get basic movement and goal-reaching working
2. **Then**: Add obstacles, multiple goals, or time pressure
3. **Finally**: Add complex dynamics, partial observability, or multi-agent interactions

### Design for Learning
- **Clear Success Criteria**: Agent should know when it's doing well
- **Reasonable Difficulty**: Not too easy (trivial) or too hard (impossible)
- **Consistent Rules**: Same action in same state should have same effect
- **Informative Observations**: Include everything needed for optimal decisions

### Think About Your Research Question
- **Navigation**: Focus on spatial reasoning and path planning
- **Control**: Emphasize dynamics, stability, and continuous actions
- **Strategy**: Include partial information, opponent modeling, or long-term planning
- **Optimization**: Design clear trade-offs and resource constraints

## Next Steps

Congratulations! You now know how to create custom RL environments. Here's what to explore next:

1. **Add rendering** to visualize your environment ([complete tutorial](../tutorials/gymnasium_basics/environment_creation))
2. **Train an agent** on your custom environment ([training guide](train_agent))
3. **Experiment with different reward functions** to see how they affect learning
4. **Try wrapper combinations** to modify your environment's behavior
5. **Create more complex environments** with obstacles, multiple agents, or continuous actions

The key to good environment design is iteration - start simple, test thoroughly, and gradually add complexity as needed for your research or application goals.
