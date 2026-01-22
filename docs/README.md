# Gymnasium-docs

这个文件夹包含了[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)的文档。

## 修改环境页面的说明

### 编辑环境页面

首先，Fork Gymnasium 并编辑环境 Python 文件中的文档字符串。然后，通过 `pip install` 安装你的 Gymnasium Fork，并在该仓库中运行 `docs/_scripts/gen_mds.py`。这将自动生成一个 Markdown 格式的环境文档文件。

### 添加新环境

确保环境已经在 Gymnasium（或你的 Fork）中。确保该环境的 Python 文件有一个正确格式的 Markdown 文档字符串。通过 `pip install -e .` 安装后，运行 `docs/_scripts/gen_mds.py`。这将自动为该环境生成一个 md 页面。然后，完成[其他步骤](https://chatgpt.com/c/6971f363-389c-832d-9551-4edf2d5037df#其他步骤)。

#### 其他步骤

- 将相应的 gif 添加到 `docs/_static/videos/{ENV_TYPE}` 文件夹中，其中 `ENV_TYPE` 是你新环境的类别（例如 mujoco）。遵循 snake_case 命名规范。或者，运行 `docs/_scripts/gen_gifs.py`。
- 编辑 `docs/environments/{ENV_TYPE}/index.md`，并将对应新环境的文件名添加到 `toctree` 中。

## 构建文档

安装所需的包和 Gymnasium（或你的 Fork）：

```
pip install gymnasium
pip install -r docs/requirements.txt
```

一次性构建文档：

```
cd docs
make dirhtml
```

每次修改后自动重建文档：

```
cd docs
sphinx-autobuild -b dirhtml --watch ../gymnasium --re-ignore "pickle$" . _build
```

然后，您可以在浏览器中打开 [http://localhost:8000](http://localhost:8000/) 来查看文档的实时更新版本。

## 编写教程

我们使用 Sphinx-Gallery 来构建 `docs/tutorials` 目录中的教程。查看 `docs/tutorials/demo.py` 以查看教程示例，更多信息请参考 [Sphinx-Gallery 文档](https://sphinx-gallery.github.io/stable/syntax.html)。

要将 Jupyter Notebooks 转换为 Python 教程，您可以使用 [这个脚本](https://gist.github.com/mgoulao/f07f5f79f6cd9a721db8a34bba0a19a7)。

如果您希望 Sphinx-Gallery 执行教程（以添加输出和图表），那么文件名应以 `run_` 开头。请注意，这会增加构建时间，因此确保脚本的执行时间不超过几秒钟。





# Gymnasium-docs

This folder contains the documentation for [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

## Instructions for modifying environment pages

### Editing an environment page

Fork Gymnasium and edit the docstring in the environment's Python file. Then, pip install your Gymnasium fork and run `docs/_scripts/gen_mds.py` in this repo. This will automatically generate a Markdown documentation file for the environment.

### Adding a new environment

Ensure the environment is in Gymnasium (or your fork). Ensure that the environment's Python file has a properly formatted markdown docstring. Install using `pip install -e .` and then run `docs/_scripts/gen_mds.py`. This will automatically generate a md page for the environment. Then complete the [other steps](#other-steps).

#### Other steps

- Add the corresponding gif into the `docs/_static/videos/{ENV_TYPE}` folder, where `ENV_TYPE` is the category of your new environment (e.g. mujoco). Follow snake_case naming convention. Alternatively, run `docs/_scripts/gen_gifs.py`.
- Edit `docs/environments/{ENV_TYPE}/index.md`, and add the name of the file corresponding to your new environment to the `toctree`.

## Build the Documentation

Install the required packages and Gymnasium (or your fork):

```
pip install gymnasium
pip install -r docs/requirements.txt
```

To build the documentation once:

```
cd docs
make dirhtml
```

To rebuild the documentation automatically every time a change is made:

```
cd docs
sphinx-autobuild -b dirhtml --watch ../gymnasium --re-ignore "pickle$" . _build
```

You can then open http://localhost:8000 in your browser to watch a live updated version of the documentation.

## Writing Tutorials

We use Sphinx-Gallery to build the tutorials inside the `docs/tutorials` directory. Check `docs/tutorials/demo.py` to see an example of a tutorial and [Sphinx-Gallery documentation](https://sphinx-gallery.github.io/stable/syntax.html) for more information.

To convert Jupyter Notebooks to the python tutorials you can use [this script](https://gist.github.com/mgoulao/f07f5f79f6cd9a721db8a34bba0a19a7).

If you want Sphinx-Gallery to execute the tutorial (which adds outputs and plots) then the file name should start with `run_`. Note that this adds to the build time so make sure the script doesn't take more than a few seconds to execute.
