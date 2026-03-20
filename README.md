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
# 升级 pip 版本
pip install --upgrade pip

# 安装 PyTorch 2.9.1 + torchvision 0.24.1（适配 CUDA 12.8）
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

## 3. 克隆仓库并安装
```bash
# 进入目标工作目录（以 ~/w_work/genesis_ws 为例）
cd ~/w_work/genesis_ws

# 克隆 Genesis 仓库（注意仓库名首字母大写）
git clone https://github.com/Genesis-Embodied-AI/Genesis.git

# 进入克隆后的仓库目录
cd Genesis

# 以开发模式安装仓库依赖
pip install -e .[dev] 
```

## 4. 验证安装
```bash
# 运行示例脚本验证环境是否搭建成功
python examples/tutorials/hello_genesis.py
```


cd ~/w_work/genesis_ws
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
pip install -e .

直接在 scripts/train.py 中调用 rsl_rl 提供的 PPO 算法类即可。




