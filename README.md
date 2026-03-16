# Human3R Stream Pipeline

在进行环境配置和运行前，请先克隆 Human3R 主仓库，并进入项目目录：


git clone [https://github.com/fanegg/Human3R.git](https://github.com/fanegg/Human3R.git)
cd Human3R



## 📁 目录结构

* `client/`: **客户端脚本（本地运行）**。负责本地摄像头的视频流采集、数据网络发送以及推理结果的本地可视化。
* `server/`: **服务端脚本（GPU 服务器运行）**。负责接收视频流，常驻加载 Human3R 模型并执行流式在线推理（`stream_server` 放置于 Human3R 主目录下）。

---

## ⚠️ 服务器环境配置与踩坑指南

在共享服务器部署 Human3R 实时推理后端时，极易遇到依赖冲突与 CUDA 算子编译报错。请严格按照以下说明配置环境。

### 1. 依赖异常处理 (Build Isolation)

**请勿直接运行** `pip install -r requirements.txt`，需要拆分处理以避免构建隔离（Build Isolation）带来的问题：

* **chumpy**: 需要从 `requirements.txt` 中移除，避开 build isolation 阶段缺乏 pip 的问题。单独通过 Git 源码安装：
    ```bash
    pip install --no-build-isolation git+[https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17](https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17)

    ```
* **gsplat**: 官方推荐的源码构建极易失败，建议直接使用 PyPI 的稳定版：
    ```bash
    pip install gsplat
    ```

### 2. CUDA 扩展编译对齐 (curope)

`curope` 编译失败会导致算子静默降级为极慢的 PyTorch 原生实现，严重拖垮流式推理的实时帧率。

* **版本对齐**: 环境内 PyTorch 的 CUDA runtime 必须与本机 `nvcc` 工具链严格匹配，否则会报 `Unsupported gpu architecture`。
* **环境变量**: 在编译前，**必须强制显式注入本地 CUDA 路径**（本机环境推荐使用 12.2）。请在终端执行以下命令：
    ```bash
    export CUDA_HOME=/usr/local/cuda-12.2
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    ```

### 3. 模型权重手动补齐

官方脚本和接口在服务器受限网络下极不稳定，强烈建议手动下载并补齐文件。

* **主权重**: 建议在本地下载好 `human3r_672S.pth` 后，通过 SSH 直传至服务器的指定目录。
* **核心依赖清单**: 即使官方脚本运行结束，仍需手动确认并补齐以下核心结构文件，否则运行时必然会报 `FileNotFoundError`：
    * `J_regressor_h36m.npy` (原脚本依赖的网盘下载常断)
    * `smplx2smpl.pkl`
    * `smpl_mean_params.npz`

---

## 🚀 快速启动

**前置提示**：由于服务端和客户端均绑定 `127.0.0.1:9999`，必须通过 **SSH 隧道**进行本地端口转发，否则客户端无法将数据发往服务器。

### Step 1: 启动服务端推理 (Server)

登录您的 GPU 服务器，激活对应的环境，并启动服务端进程：

```bash
# 进入服务端目录
cd server

# 指定显卡并启动服务
CUDA_VISIBLE_DEVICES=7 python stream_server.py
```

### Step 2: 建立 SSH 端口转发 (Local)

在您的**本地电脑**上，新建一个终端窗口，将本地的 9999 端口映射到服务器的 9999 端口：

```bash
# 请将 "用户名" 替换为您实际的服务器登录名称
ssh -N -f -L 9999:127.0.0.1:9999 用户名@服务器
```

### Step 3: 启动客户端采集 (Client)

在您的**本地电脑**上，开启摄像头推流及可视化客户端：

```bash
# 进入客户端目录
cd client

# 启动客户端
python stream_client_vis.py
```


