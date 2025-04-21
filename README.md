# FCOS-cuda113
🚀 FCOS 编译指南（原版 CUDA 10.2 环境）
1. 环境配置
推荐环境：

bash
复制
编辑
Python: 3.7
PyTorch: 1.10.0
TorchVision: 0.11.0
CUDA: 10.2
安装方式（conda）：

bash
复制
编辑
conda create -n fcos_py37 python=3.7 -y
conda activate fcos_py37
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
2. 遇到的常见问题
ImportError: cannot import name '_C'
text
复制
编辑
from fcos_core import _C
ImportError: cannot import name '_C'
💡 解决方法：

bash
复制
编辑
python setup.py clean
python setup.py build_ext
python setup.py develop
CUDA mismatch 问题
pgsql
复制
编辑
The detected CUDA version (11.3) mismatches the version that was used to compile PyTorch (10.2)
设置 CUDA_HOME 环境变量（仅在 cmd.exe 中有效）：

bat
复制
编辑
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
检查是否设置了多个 CUDA 路径，优先确保 PATH 中 CUDA 10.2 排在前面

编译报错（如 AT_CHECK）
text
复制
编辑
error: ‘AT_CHECK’ was not declared in this scope
🔧 原因：AT_CHECK 已废弃于高版本，替换为：

cpp
复制
编辑
TORCH_CHECK(...)
可以使用正则批量替换所有 .cpp/.cu 文件中的 AT_CHECK → TORCH_CHECK

3. 编译建议
推荐使用 python setup.py develop 而非 install

不建议开启 ninja，可能导致 .obj 文件链接失败：

bash
复制
编辑
set USE_NINJA=0
⚙️ FCOS 修改版本适配 CUDA 11.3 + PyTorch 1.11.0
1. 环境配置
推荐环境：

bash
复制
编辑
Python: 3.8 or 3.9
PyTorch: 1.11.0
CUDA: 11.3
安装命令：

bash
复制
编辑
conda create -n fcos113 python=3.8 -y
conda activate fcos113
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
2. 修改源码以适配高版本 CUDA/PyTorch
2.1 移除已废弃的头文件
cpp
复制
编辑
// 删除
#include <THC/THC.h>

// 替换 THCudaCheck
AT_CUDA_CHECK(cudaGetLastError());
2.2 替换 THCCeilDiv
cpp
复制
编辑
#include <ATen/ceil_div.h>
int blocks = at::ceil_div(total_elements, threads_per_block);
2.3 处理 atomicAdd 不支持 c10::Half 错误
错误：

text
复制
编辑
error: no instance of overloaded function "atomicAdd" matches the argument list (c10::Half *, c10::Half)
解决方法（暂不完美）：

替换为支持 half 类型的 gpuAtomicAdd（需编写或启用相关 CUDA intrinsics）

或强制转 float：

cpp
复制
编辑
atomicAdd(reinterpret_cast<float*>(ptr), static_cast<float>(val));
注意：这部分兼容性复杂，最好使用 float 版本运行验证

3. 避坑指南
清理旧的编译缓存：

bash
复制
编辑
python setup.py clean
rm -rf build/
Ninja 编译器相关问题：

禁用 ninja，否则可能导致 .obj 缺失问题

若必须使用，确保 ninja 与 torch 和 cuda 完全兼容

4. 调试小技巧
PyCharm 中断点需设置在 import 附近

检查 torch.version.cuda 输出是否与 nvcc -V 一致

若显卡如 RTX 3060 不被当前 torch 支持，需安装支持 sm_86 的 PyTorch 构建
