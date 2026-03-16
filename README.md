##Human3R Stream Pipeline

#快速启动

由于服务端和客户端均绑定 127.0.0.1:9999，必须通过 SSH 隧道进行本地端口转发。

1. 启动服务端推理 (Server)
在 GPU 服务器 (10.28.0.105) 激活环境并启动服务：
stream_server放置于Human3R主目录下

cd server
CUDA_VISIBLE_DEVICES=7 python stream_server.py


2. 建立 SSH 端口转发 (Local)
在本地电脑新建一个终端，将本地 9999 端口映射到服务器（请将 用户名 替换为实际名称）：

ssh -N -f -L 9999:127.0.0.1:9999 用户名@10.28.0.105

3. 启动客户端采集 (Client)
在本地电脑启动摄像头推流：

cd client
python stream_client_vis.py

📁 目录结构

client/: 客户端脚本（本地运行），负责摄像头流采集、数据发送及本地结果可视化。

server/: 服务端脚本（GPU 服务器运行），负责接收视频流、常驻加载 Human3R 模型并执行流式在线推理。

⚠️ 服务器环境配置踩坑记录

记录在共享服务器部署 Human3R 实时推理后端时的核心问题，重点解决依赖冲突与 CUDA 算子编译报错。

1. 依赖异常处理 (Build Isolation)

不可直接 pip install -r requirements.txt，需拆分处理：

chumpy: 从 txt 中移除，避开 build isolation 阶段缺乏 pip 的问题，单独安装：

pip install --no-build-isolation git+[https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17](https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17)


gsplat: 官方推荐的源码构建极易失败，直接使用 PyPI 稳定版：

pip install gsplat


2. CUDA 扩展编译对齐 (curope)

curope 编译失败会导致算子静默降级为极慢的 PyTorch 原生实现，严重拖垮流式推理的实时帧率。

版本对齐: 环境内 PyTorch 的 CUDA runtime 必须与本机 nvcc 工具链严格匹配，否则报 Unsupported gpu architecture。

环境变量: 编译前必须强制显式注入本地 CUDA 路径（本机推荐 12.2）：

export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


3. 模型权重手动补齐

官方 fetch_smplx.sh 脚本和 HF 接口在服务器受限网络下极不稳定。

主权重: 建议本地下载 human3r_672S.pth 后，通过 SSH 直传至服务器指定目录。

核心依赖清单: 即使脚本跑完，仍需手动寻找并补齐以下核心结构文件，否则运行必报 FileNotFoundError：

J_regressor_h36m.npy (原脚本依赖的 Google Drive 下载常断)

smplx2smpl.pkl (同上)

smpl_mean_params.npz (同上)


