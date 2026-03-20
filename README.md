# genesis_ws 环境搭建指南

## 1. 创建并激活 Conda 环境
```bash
# 创建名为 genesis-env 的 Python 3.11 环境
conda create -n genesis-env python=3.11 -y

# 激活 genesis-env 环境
conda activate genesis-env
```

## 2. 安装 PyTorch 及依赖
```bash
# 升级 pip 版本，避免依赖安装兼容问题
pip install --upgrade pip

# 安装 PyTorch 2.9.1 + torchvision 0.24.1（适配 CUDA 12.8）
# 若本地 CUDA 版本非 12.8，需替换为对应版本的 PyTorch 安装指令
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

## 3. 克隆 Genesis 仓库并安装核心依赖
```bash
# 进入目标工作目录（以 ~/w_work/genesis_ws 为例）
cd ~/w_work/genesis_ws

# 克隆 Genesis 仓库
git clone https://github.com/Genesis-Embodied-AI/Genesis.git

# 进入克隆后的仓库目录
cd Genesis

# 以开发模式安装仓库依赖
pip install -e .[dev] 
```

## 4. 安装 rsl_rl 强化学习库
```bash
# 回到 genesis_ws 根目录
cd ~/w_work/genesis_ws

# 克隆 rsl_rl 仓库
git clone https://github.com/leggedrobotics/rsl_rl.git

# 进入 rsl_rl 目录并以开发模式安装
cd rsl_rl
pip install -e .
```

## 5. 验证环境安装
```bash
# 运行 Genesis 示例脚本验证核心环境是否搭建成功
python ~/w_work/genesis_ws/Genesis/examples/tutorials/hello_genesis.py

# 验证 rsl_rl 可调用性
python -c "from rsl_rl.algorithms import PPO; print('rsl_rl PPO 算法导入成功')"
```

