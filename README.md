[![Python](https://img.shields.io/pypi/pyversions/gymnasium.svg)](https://badge.fury.io/py/gymnasium)
[![PyPI](https://badge.fury.io/py/gymnasium.svg)](https://badge.fury.io/py/gymnasium)
[![arXiv](https://img.shields.io/badge/arXiv-2407.17032-b31b1b.svg)](https://arxiv.org/abs/2407.17032)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/github/license/Farama-Foundation/Gymnasium)](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
    <a href="https://gymnasium.farama.org/" target = "_blank">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium-text.png" width="500px" />
</a>



</p>

Gymnasium 是一个开源 Python 库，用于开发和比较强化学习算法。它提供了一个标准的 API，用于在学习算法和环境之间进行通信，并且提供了一组符合该 API 的标准环境。这是 OpenAI 的 [Gym](https://github.com/openai/gym) 库的一个分支，OpenAI 已将维护工作交给了外部团队，未来的维护将在此分支上进行。

文档网站位于 [gymnasium.farama.org](https://gymnasium.farama.org/)，我们还有一个公共的 Discord 服务器（我们也用它来协调开发工作），你可以在此加入：https://discord.gg/bnJ6kubTg6

## 环境a

Gymnasium 包括以下几类环境，并且提供了许多第三方环境：

- [经典控制](https://gymnasium.farama.org/environments/classic_control/) - 这些是基于现实世界问题和物理的经典强化学习环境。
- [Box2D](https://gymnasium.farama.org/environments/box2d/) - 这些环境涉及基于物理控制的玩具游戏，使用基于 box2d 的物理引擎和基于 PyGame 的渲染。
- [Toy Text](https://gymnasium.farama.org/environments/toy_text/) - 这些环境设计得非常简单，具有小的离散状态和动作空间，因此容易学习。由于这一特点，它们适合用于调试强化学习算法的实现。
- [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) - 基于物理引擎的环境，支持多关节控制，比 Box2D 环境更为复杂。
- [Atari](https://ale.farama.org/) - 模拟 Atari 2600 ROM 的模拟器，具有较高的复杂性，适合代理进行学习。
- [第三方](https://gymnasium.farama.org/environments/third_party_environments/) - 一些与 Gymnasium API 兼容的环境已经被创建。请注意软件创建时的版本，并在必要时使用 `gymnasium.make` 中的 `apply_env_compatibility`。

## 安装

要安装基础的 Gymnasium 库，使用 `pip install gymnasium`

这不包括所有环境类别的依赖项（因为有大量环境，而且某些依赖项在特定系统上可能会出现安装问题）。你可以通过安装某个环境类别的依赖项，如 `pip install "gymnasium[atari]"`，或者使用 `pip install "gymnasium[all]"` 来安装所有依赖项。

我们支持并测试 Linux 和 macOS 上的 Python 3.10、3.11、3.12 和 3.13。我们会接受与 Windows 相关的 PR，但不官方支持 Windows。

## API

Gymnasium API 将环境建模为简单的 Python `env` 类。创建环境实例并与之交互非常简单——以下是使用 "CartPole-v1" 环境的示例：

```python
import gymnasium as gym
env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

## 相关的著名库

请注意，这只是一个不完全的列表，仅包括在新手询问推荐时，维护者最常指向的库。

- [CleanRL](https://github.com/vwxyzjn/cleanrl) 是一个基于 Gymnasium API 的学习库。它旨在面向新手，并提供了非常好的参考实现。
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) 是 Gymnasium 的多智能体版本，包含了多个实现的环境，例如多智能体的 Atari 环境。
- Farama 基金会还维护了许多其他的 [环境](https://farama.org/projects)，这些环境由与 Gymnasium 同一团队维护，并使用 Gymnasium API。

## 环境版本控制

Gymnasium 为了可重复性，保持严格的版本控制。所有环境的名称都以类似 "-v0" 的后缀结尾。当对环境进行更改可能影响学习结果时，版本号会增加1，以防止潜在的混淆。这些版本控制继承自 Gym。

## 贡献

我们欢迎社区的贡献！
请查看我们的 [CONTRIBUTING.md](https://github.com/Farama-Foundation/Gymnasium/blob/main/CONTRIBUTING.md)，了解如何开始贡献。

## 支持 Gymnasium 的开发

如果您经济上有能力，并且希望支持 Gymnasium 的开发，请与社区中的其他人一起 [捐赠给我们](https://github.com/sponsors/Farama-Foundation)。

## 引用

您可以使用我们的相关论文（https://arxiv.org/abs/2407.17032）引用 Gymnasium：

```
@article{towers2024gymnasium,
  title={Gymnasium: A Standard Interface for Reinforcement Learning Environments},
  author={Towers, Mark and Kwiatkowski, Ariel and Terry, Jordan and Balis, John U and De Cola, Gianluca and Deleu, Tristan and Goul{\~a}o, Manuel and Kallinteris, Andreas and Krimmel, Markus and KG, Arjun and others},
  journal={arXiv preprint arXiv:2407.17032},
  year={2024}
}
```

## 仓库赞助商

<h3 style="margin-bottom:10;margin-top:0"><a href="https://ref.wisprflow.ai/UnmiceG">Wispr Flow</a></h3>

<a href="https://ref.wisprflow.ai/UnmiceG">
  <img src="assets/wispr-flow.svg" alt="Wispr Flow" width="100">
</a>

<h3 style="margin-bottom:10;margin-top:0">Dictation that understands code</h3>
<h4 style="margin-top:0;">Ship 4x faster with developer-first dictation that works in every app.</h4>

<p style="margin-top:50;">If you'd like to sponsor Gymnasium or other Farama repositories and have your logo here, <a href="mailto:contact@farama.org">contact us</a>.</p>



## ----------------------------------



[![Python](https://img.shields.io/pypi/pyversions/gymnasium.svg)](https://badge.fury.io/py/gymnasium)
[![PyPI](https://badge.fury.io/py/gymnasium.svg)](https://badge.fury.io/py/gymnasium)
[![arXiv](https://img.shields.io/badge/arXiv-2407.17032-b31b1b.svg)](https://arxiv.org/abs/2407.17032)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/github/license/Farama-Foundation/Gymnasium)](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
    <a href="https://gymnasium.farama.org/" target = "_blank">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium-text.png" width="500px" />
</a>


</p>

Gymnasium is an open source Python library for developing and comparing reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments, as well as a standard set of environments compliant with that API. This is a fork of OpenAI's [Gym](https://github.com/openai/gym) library by its maintainers (OpenAI handed over maintenance a few years ago to an outside team), and is where future maintenance will occur going forward.

The documentation website is at [gymnasium.farama.org](https://gymnasium.farama.org), and we have a public discord server (which we also use to coordinate development work) that you can join here: https://discord.gg/bnJ6kubTg6

## Environments

Gymnasium includes the following families of environments along with a wide variety of third-party environments

* [Classic Control](https://gymnasium.farama.org/environments/classic_control/) - These are classic reinforcement learning based on real-world problems and physics.
* [Box2D](https://gymnasium.farama.org/environments/box2d/) - These environments all involve toy games based around physics control, using box2d based physics and PyGame-based rendering
* [Toy Text](https://gymnasium.farama.org/environments/toy_text/) - These environments are designed to be extremely simple, with small discrete state and action spaces, and hence easy to learn. As a result, they are suitable for debugging implementations of reinforcement learning algorithms.
* [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) - A physics engine based environments with multi-joint control which are more complex than the Box2D environments.
* [Atari](https://ale.farama.org/) - Emulator of Atari 2600 ROMs simulated that have a high range of complexity for agents to learn.
* [Third-party](https://gymnasium.farama.org/environments/third_party_environments/) - A number of environments have been created that are compatible with the Gymnasium API. Be aware of the version that the software was created for and use the `apply_env_compatibility` in `gymnasium.make` if necessary.

## Installation

To install the base Gymnasium library, use `pip install gymnasium`

This does not include dependencies for all families of environments (there's a massive number, and some can be problematic to install on certain systems). You can install these dependencies for one family like `pip install "gymnasium[atari]"` or use `pip install "gymnasium[all]"` to install all dependencies.

We support and test for Python 3.10, 3.11, 3.12 and 3.13 on Linux and macOS. We will accept PRs related to Windows, but do not officially support it.

## API

The Gymnasium API models environments as simple Python `env` classes. Creating environment instances and interacting with them is very simple- here's an example using the "CartPole-v1" environment:

```python
import gymnasium as gym
env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

## Notable Related Libraries

Please note that this is an incomplete list, and just includes libraries that the maintainers most commonly point newcomers to when asked for recommendations.

* [CleanRL](https://github.com/vwxyzjn/cleanrl) is a learning library based on the Gymnasium API. It is designed to cater to newer people in the field and provides very good reference implementations.
* [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) is a multi-agent version of Gymnasium with a number of implemented environments, i.e. multi-agent Atari environments.
* The Farama Foundation also has a collection of many other [environments](https://farama.org/projects) that are maintained by the same team as Gymnasium and use the Gymnasium API.

## Environment Versioning

Gymnasium keeps strict versioning for reproducibility reasons. All environments end in a suffix like "-v0".  When changes are made to environments that might impact learning results, the number is increased by one to prevent potential confusion. These were inherited from Gym.

## Contributing

We welcome contributions from the community!
Please see our [CONTRIBUTING.md](https://github.com/Farama-Foundation/Gymnasium/blob/main/CONTRIBUTING.md) for details on how to get started.

## Support Gymnasium's Development

If you are financially able to do so and would like to support the development of Gymnasium, please join others in the community in [donating to us](https://github.com/sponsors/Farama-Foundation).

## Citation

You can cite Gymnasium using our related paper (https://arxiv.org/abs/2407.17032) as:

```
@article{towers2024gymnasium,
  title={Gymnasium: A Standard Interface for Reinforcement Learning Environments},
  author={Towers, Mark and Kwiatkowski, Ariel and Terry, Jordan and Balis, John U and De Cola, Gianluca and Deleu, Tristan and Goul{\~a}o, Manuel and Kallinteris, Andreas and Krimmel, Markus and KG, Arjun and others},
  journal={arXiv preprint arXiv:2407.17032},
  year={2024}
}
```

## Repository Sponsors

<h3 style="margin-bottom:10;margin-top:0"><a href="https://ref.wisprflow.ai/UnmiceG">Wispr Flow</a></h3>

<a href="https://ref.wisprflow.ai/UnmiceG">
  <img src="assets/wispr-flow.svg" alt="Wispr Flow" width="100">
</a>

<h3 style="margin-bottom:10;margin-top:0">Dictation that understands code</h3>
<h4 style="margin-top:0;">Ship 4x faster with developer-first dictation that works in every app.</h4>

<p style="margin-top:50;">If you'd like to sponsor Gymnasium or other Farama repositories and have your logo here, <a href="mailto:contact@farama.org">contact us</a>.</p>
