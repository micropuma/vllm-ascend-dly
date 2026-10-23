# vLLM Ascend 源码构建

本文记录在远端 A3 开发容器中使用 vLLM Ascend 源码的构建方法。

## 固定版本

镜像不是通过 `git clone` 获取，而是从 NJU registry 使用 Docker 拉取。源码单独从 Git 仓库获取并挂载到容器中。

```bash
docker pull quay.nju.edu.cn/atlas-ci/vllm-ascend:v0.28.0-a3
```

本次远端实际使用的镜像 digest 是：

```text
quay.nju.edu.cn/atlas-ci/vllm-ascend@sha256:bded7da4f300800ec4715dbde44531c4735cf0950fa313977248ef0dd74866b8
```

为避免 tag 被覆盖，生产复现时可以直接使用 digest：

```bash
docker pull quay.nju.edu.cn/atlas-ci/vllm-ascend@sha256:bded7da4f300800ec4715dbde44531c4735cf0950fa313977248ef0dd74866b8
```

源码固定为 Git commit：

```text
vllm-ascend: 3caf228dd8667a525809e92b887a809a88ce17ff
vLLM release: v0.28.0
vLLM verified commit: b2f685834a6456197e7033966fdef52a23f1abcd
```

源码部署到远端后，必须确认：

```bash
git reset --hard 3caf228dd8667a525809e92b887a809a88ce17ff
git clean -fd
```

## 固定环境

使用已经准备好的容器和固定软件栈，不需要另外创建 Python 虚拟环境：

```text
容器: vllm-ascend-dly-v028
镜像: quay.nju.edu.cn/atlas-ci/vllm-ascend:v0.28.0-a3
源码: /workspace/vllm-ascend-dly-git
Python: /usr/local/python3.12.13/bin/python3
vLLM: 0.28.0
PyTorch: 2.10.0+cpu
Torch-NPU: 2.10.0.post4
CANN: 9.1.0
Triton Ascend: 3.2.2
```

镜像中的 vLLM、PyTorch、Torch-NPU、CANN 和 Triton Ascend 已经按兼容组合准备好。不要在此流程中重新安装或升级这些依赖。

检查镜像是否为固定 digest：

```bash
docker image inspect quay.nju.edu.cn/atlas-ci/vllm-ascend:v0.28.0-a3 \
    --format '{{json .RepoDigests}}'
```

## 创建并启动容器

容器只在首次创建、或需要修改挂载时用下面的 `docker run` 创建一次。它只跑
`sleep infinity`，不随命令退出而销毁，适合反复 `docker exec` 进入：

```bash
IMAGE=quay.nju.edu.cn/atlas-ci/vllm-ascend:v0.28.0-a3
NAME=vllm-ascend-dly-v028
SRC=/public-flash/chenxi/douliyang/llm-inference-workspace/vllm-ascend-dly-git

docker run -d --name "$NAME" \
  --init \
  --restart unless-stopped \
  --net=host \
  --privileged=true \
  --shm-size=2g \
  --workdir /workspace/vllm-ascend-dly-git \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  $(for i in $(seq 0 15); do printf -- '--device /dev/davinci%s ' "$i"; done) \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$SRC:/workspace/vllm-ascend-dly-git" \
  -v /root/.cache:/root/.cache \
  -v /data/weight:/data/weight \
  -v /root/.modelscope:/root/.modelscope \
  "$IMAGE" sleep infinity
```

A3 是双 DIE：宿主 8 个 `/dev/davinci0..7` 对应容器内 16 个 `torch.npu` 逻辑设备，
因此必须按 `davinci0..15` 展开映射；`davinci_manager`、`devmm_svm`、`hisi_hdc`
是管理节点，缺一不可。`--shm-size=2g` 提供给 vLLM 的共享内存。

> **`--init` 必须有。** 容器主进程是 `sleep infinity`，不加 `--init` 时**它就是 PID 1**；
> `sleep` 从不调用 `wait()`，于是任何在容器里死掉/变孤儿的子进程都**永久滞留在 `Z`（僵尸）态**
> —— 实测跑几轮 vLLM + 调试后宿主 `ps` 攒了 **5861 个 Z**（≈几十 MB 内核内存 + 等量 PID 槽，
> 且**无上界**）。加了 `--init` 后 docker 会注入 `docker-init`（tini）当 PID 1，它负责
> ①收尸（`wait()` 循环）②转发 `SIGTERM/SIGINT`。**对使用方式零影响**：
> `docker exec` 进出、VS Code Remote-SSH attach、集成终端全都照旧
> （`docker exec` 起的进程由 containerd-shim 收尸，本来就不经过 PID 1）。
> 已经存在且没带 `--init` 的容器，`docker restart` 也能清空僵尸（PID 命名空间重建，
> **保留可写层**）；要根治只能下次重建时带上 `--init`。
>
> **给已在跑、又不想丢掉可写层的容器补 `--init`**（2026-09-19 就这么做的）：
> `docker commit` 把可写层冻成本地镜像 → `docker rm` 旧容器 → 用**同一批参数 + `--init`**
> 从新镜像重起。可写层大小决定代价（本容器 1.07GB，几秒）。
>
> ⚠️ 走这条路之后，**当前 `vllm-ascend-dly-v028` 是从本地镜像
> `vllm-ascend-dly:v028-9-19-init`（= quay 基础镜像 + 1.07GB 可写层）起的，不再是
> `quay.nju.edu.cn/atlas-ci/vllm-ascend:v0.28.0-a3` 本身**。要回到纯 quay 镜像，
> 按上面的 `docker run`（带 `--init`）重建，再按下面「可写层会丢什么」那段补回。

> **必须用 `--net=host` + `--privileged=true`。** 默认 bridge 网络下容器拿到的是
> `172.17.0.3` 这类隔离 IP；多 DP 组拉起时各自要在宿主 IP 上建 HCCL 端点，会与宿主
> 网络命名空间错配，服务在模型加载完约 3 分钟后报
> `Communication_Error_Bind_IP_Port(EI0019)` 崩溃。`--net=host` 让容器直接复用宿主
> 网络栈，`--privileged=true` 是配套权限（参照同机已跑通容器的做法）。
>
> 切到 host 网络后若仍报 `... port 60000 have already been bound`，那是 DP2 两个
> DP 组抢同一 HCCL base 端口的另一类问题，用容器内环境变量
> `HCCL_IF_BASE_PORT=61000` 解决，详见
> [`model-serving/README.md`](model-serving/README.md) 第 3 节。


下面这些 `-v` 都是 **bind mount**：容器内外看到的是同一份文件，任何一侧的改动
另一侧立即可见（不是拷贝，也没有同步延迟）：

| 宿主 | 容器 | 用途 |
| --- | --- | --- |
| `$SRC` | `/workspace/vllm-ascend-dly-git` | vLLM Ascend 源码，容器内改动直接落盘 |
| `/data/weight` | `/data/weight` | 量化模型权重；宿主新增模型容器内立即可见 |
| `/root/.cache` | `/root/.cache` | ModelScope / pip 缓存 |
| `/root/.modelscope` | `/root/.modelscope` | ModelScope 登录凭据（不挂会随容器重建丢失）|
| `/usr/local/Ascend/driver`、`/usr/local/Ascend/firmware` | 同路径 | 宿主驱动 / 固件 |
| `/usr/local/dcmi` | 同路径 | DCMI 管理库 |
| `/usr/local/bin/npu-smi`、`/etc/ascend_install.info` | 同路径 | NPU 监控与安装信息 |

停止后重新启动、以及进入容器：

```bash
docker start vllm-ascend-dly-v028
docker exec -it vllm-ascend-dly-v028 bash
```

> Docker 不支持给正在运行的容器增删挂载。要改挂载必须 `docker rm` 后用上面的
> `docker run` 重建。源码、模型、缓存都在挂载盘上，重建不会丢失。
>
> **但「可写层」会丢**（bind mount 之外的那一层）：`_C_ascend` 编译产物与源码同在挂载盘上
> （安全），**丢的是** editable 安装元数据（`.pth`/`dist-info`）与 `debugpy`。补法见
> [`model-serving/debug/README.md`](model-serving/debug/README.md) 的「容器重建之后要补什么」。
> （调试全流程另见本地文档 `tutorial/inference/vllm-ascend/docker-debug.md`，不在本仓库内。）
> 只是清僵尸（不重建）的话 `docker restart` 就够，可写层保留。

## 源码构建

### 方式一：仅 Python 源码开发（不编算子，快）

只改 Python 代码、不需要 `torch.ops._C_ascend.*` 时用这个模式（几分钟）：

```bash
docker exec -it vllm-ascend-dly-v028 bash
cd /workspace/vllm-ascend-dly-git

export SOC_VERSION=ascend910_9391
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export COMPILE_CUSTOM_KERNELS=0      # ← 不编算子：不会生成 vllm_ascend_C.so
export MAX_JOBS=8

python3 -m pip install \
    --no-build-isolation \
    --no-deps \
    -e .
```

> ⚠️ **`COMPILE_CUSTOM_KERNELS=0` 不会生成 `vllm_ascend_C.so`，`vllm_ascend/_cann_ops_custom/` 也是空的。**
> 这种安装只能跑 Python 层；一旦 `vllm serve` 走到 warmup，就会报
> `'_OpNamespace' '_C_ascend' object has no attribute 'moe_gating_top_k'`。
> **要真把模型跑起来，必须用下面的「方式二」重建。**

`-e .` 会让 Python 直接使用当前源码目录，适合源码开发；修改 Python 文件后无需重新安装。`--no-deps` 防止 pip 改动镜像中已经验证的依赖，`--no-build-isolation` 使用镜像已有的编译工具和 CANN 环境。

镜像内自带一份 editable 的 `vllm_ascend 0.0.0`（指向镜像里的 `/vllm-workspace/vllm-ascend`）。
上面的 `-e .` 会先把它卸载，再把包路径改指到 `/workspace/vllm-ascend-dly-git`。安装后确认：

```bash
python3 -m pip show vllm-ascend | grep -E '^(Version|Editable)'
# Version: 0.1.dev5330+g3caf228dd        ← 由 git commit 推导，应与 git HEAD 一致
# Editable project location: /workspace/vllm-ascend-dly-git
```

### 方式二：完整构建（编译 `_C_ascend` 自定义算子）

`torch.ops._C_ascend.*`（`-git` 树共 **80** 个算子，含 GLM / DeepSeek 用到的 `moe_gating_top_k`、
`mla_preprocess`、`grouped_matmul_swiglu_quant*` 等）由一个 PyTorch 扩展
`vllm_ascend_C.so` 提供，它包装 `csrc/` 下编出的 45 个 CANN ACLNN 内核。
**镜像默认不带这些产物，必须自己编一次。**

#### 前置依赖 1：catlass 子模块（版本必须匹配）

> **`-git` 仓库（commit `3caf228dd`）已随仓库带上该子模块**：宿主上
> `git -C $SRC submodule status csrc/third_party/catlass` 输出
> `41bf90da655bba3c66d0acd7e00abe33960ecfd6`，因此**本节的手工拷贝步骤可以跳过**。
> 只有从**不含子模块**的历史仓库（如 `bd99c348c`）构建时才需要下面的步骤。

`csrc/build_aclnn.sh` 依赖 `csrc/third_party/catlass`（Ascend C 模板库）。它是 git
submodule，历史仓库里是空的，容器又无外网，必须先手工放进去。版本要与 `.gitmodules`
锁定的一致，否则可能编不过：

```text
csrc/third_party/catlass
  commit 41bf90da655bba3c66d0acd7e00abe33960ecfd6
  来源   https://gitcode.com/cann/catlass.git  branch master
```

远端已有一份与该 commit **完全一致**的干净副本，直接拷 `include/` 即可
（容器内外 bind mount 同一目录，在宿主执行即可）：

```bash
CATLASS_SRC=/public-flash/zhiyuan/liyuzhuo/flash-linear-attention-npu/third_party/catlass
DST=/public-flash/chenxi/douliyang/llm-inference-workspace/vllm-ascend-dly-git/csrc/third_party/catlass

cd "$CATLASS_SRC" && git rev-parse HEAD     # 必须输出 41bf90da655bba3c66d0acd7e00abe33960ecfd6
mkdir -p "$DST"
cp -a "$CATLASS_SRC/include" "$DST/"        # 只需头文件目录
```

> `build_aclnn.sh` 只检查 `csrc/third_party/catlass/include` 是否存在；不存在时会尝试
> `git submodule update --init --recursive`，因无外网必然失败。

#### 前置依赖 2：给构建走代理

CMake 的 `FetchContent` 要从 gitcode 拉 `makeself`、`abseil-cpp`、`protobuf`、`gtest`、
`json`、`secure_c`；容器没有直接外网，必须借宿主代理。

```bash
export http_proxy=http://10.6.212.42:2080
export https_proxy=$http_proxy
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,::1
```

> 单测代理是否通：`curl -sI https://gitcode.com` 应回 200。
> 若宿主代理地址变了，改这里的 `IP:PORT` 即可。

#### 构建命令

```bash
docker exec -it vllm-ascend-dly-v028 bash
cd /workspace/vllm-ascend-dly-git

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# ← 这里补上上面的代理 export

export SOC_VERSION=ascend910_9391      # A3；对应 ACLNN 构建分支 ascend910_93
export COMPILE_CUSTOM_KERNELS=1        # ← 关键：编算子
export MAX_JOBS=64

python3 -m pip install -e . --no-build-isolation --no-deps
```

`pip install` 会顺序驱动两阶段（见 `setup.py` 的 `cmake_build_ext.run()`）：

1. `run_command("build_aclnn")` → `bash csrc/build_aclnn.sh`：编 45 个 CANN 内核，装到
   `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/`；
2. CMake 编扩展 → `vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`。

全量约 20 分钟（`MAX_JOBS` 越大越快）。建议**后台跑 + 日志落盘**，避免 SSH 断线中断：

```bash
cd /workspace/vllm-ascend-dly-git
LOG=/workspace/vllm-ascend-dly-git/build_custom_ops.log
setsid nohup bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh
  export http_proxy=http://10.6.212.42:2080 https_proxy=http://10.6.212.42:2080
  export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy no_proxy=localhost,127.0.0.1,::1
  export SOC_VERSION=ascend910_9391 COMPILE_CUSTOM_KERNELS=1 MAX_JOBS=64
  python3 -m pip install -e . --no-build-isolation --no-deps
  echo "### pip exit=$?"
' >> "$LOG" 2>&1 < /dev/null &

tail -f "$LOG"
```

构建产物校验：

```bash
cd /workspace/vllm-ascend-dly-git
ls vllm_ascend/vllm_ascend_C*.so                     # 扩展存在
ls vllm_ascend/_cann_ops_custom/vendors/             # custom_transformer 存在
grep -nE 'pip exit=|build end' build_custom_ops.log  # 应看到 pip exit=0
```

## 使用自定义算子（关键）

编出来只是第一步：**`.so` 必须被显式 import，`_C_ascend` 的算子才会注册进 torch**。
`import vllm_ascend` 本身**不会**加载它。正确用法是调官方加载函数：

```python
import torch
import torch_npu                      # 必须先 import：PrivateUse1 后端由 torch_npu 注册

from vllm_ascend.utils import enable_custom_op
assert enable_custom_op() is True     # 内部会 import vllm_ascend.vllm_ascend_C

# 之后业务代码里的 torch.ops._C_ascend.* 才可用
op = torch.ops._C_ascend.moe_gating_top_k
```

服务化时这条路径由 vLLM Ascend 自己在 worker 初始化时触发
（`vllm_ascend/utils.py:enable_custom_op`），**前提是 `.so` 已编译存在**；否则
`enable_custom_op()` 会捕获 `ImportError` 返回 `False`，把所有自定义算子静默禁用，
直到某处访问 `_C_ascend.<op>` 才抛 `AttributeError`。

### 校验算子是否真的注册（不要用 `dir()`！）

`torch.ops._C_ascend` 是惰性对象，**命名空间为空时 `dir()` 也返回 `['name']`**，
`hasattr(...)` 同样不可靠，会误判。**唯一权威判据**是 torch 的算子注册表：

```bash
cd /workspace/vllm-ascend-dly-git
python3 - <<'PY'
import torch, torch_npu
from vllm_ascend.utils import enable_custom_op
print("enable_custom_op:", enable_custom_op())                 # 期望 True
names = [s.name for s in torch._C._jit_get_all_schemas()
         if s.name.startswith("_C_ascend::")]
print("registered _C_ascend ops:", len(names))                 # 期望 80
print("moe_gating_top_k:", "_C_ascend::moe_gating_top_k" in names)
print("npu_causal_conv1d_custom:", "_C_ascend::npu_causal_conv1d_custom" in names)
PY
```

## 构建验证

```bash
cd /workspace/vllm-ascend-dly-git
python3 - <<'PY'
import torch
import vllm
import vllm_ascend

print("vLLM:", vllm.__version__)
print("vLLM Ascend:", vllm_ascend.__file__)
print("NPU available:", torch.npu.is_available())
assert vllm.__version__.startswith("0.28.")
assert torch.npu.is_available()
PY
```

应看到 vLLM `0.28.0`，并且 `NPU available: True`。当前源码提交可用以下命令确认：

```bash
git log -1 --oneline
cat .github/vllm-release-tag.commit
```

## ModelScope 烟测

模型下载只给当前命令设置代理，不修改 Docker daemon，也不影响其他容器：

```bash
export VLLM_USE_MODELSCOPE=True
export ASCEND_RT_VISIBLE_DEVICES=2
export HTTP_PROXY=http://10.6.212.42:2080
export HTTPS_PROXY=http://10.6.212.42:2080

python3 tutorial/models/modelscope_qwen3_0_6b.py
```

模型会缓存到容器挂载的 ModelScope 缓存目录。若下载中断，重新执行同一命令即可续用已有缓存。


## 在线 serving

构建完成、算子注册正常后即可拉起在线服务。`GLM-5.2-w4a8c8` 的完整配方、
两套并行配置（DP2TP8+EP vs DP1TP16 关 EP）的实测对比，以及本机必须的
`HCCL_IF_BASE_PORT` 端口修复，见
[`model-serving/README.md`](model-serving/README.md)。
