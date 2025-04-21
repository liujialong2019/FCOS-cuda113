1、`torch.version.cuda` 是 PyTorch 中的一个属性，用于显示当前 PyTorch 安装所使用的 CUDA 版本。这个信息非常有用，因为它可以帮助您确认 PyTorch 是否正确地与 CUDA 集成，以及使用的是哪个 CUDA 版本。让我详细解释一下：

1. 含义：

   - 这个属性返回一个字符串，表示 PyTorch 编译时使用的 CUDA 版本。

   - 例如，如果返回 "11.7"，意味着 PyTorch 是用 CUDA 11.7 编译的。

   - ```
     import torch
     print(torch.version.cuda)
     ```

     

2、    

```
from fcos_core import _C
ImportError: cannot import name '_C' from 'fcos_core' (C:\Ob_De\2024\FOCS\FCOS-master01\FCOS-master\fcos_core\__init__.py)
```

python setup.py develop

3、需要两个版本相近

```
The detected CUDA version (10.2) mismatches the version that was used to compile
PyTorch (11.3). Please make sure to use the same CUDA versions.
```

4、最开始编译FCOS的虚拟环境是py3.7+

```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
```

5、对于 Windows PowerShell 其无法正常地去运行脚本，只有在 cmd.exe 命令中才可以, 非也非也

```
C:\Users\liujialong\.conda\envs\Co-DETR-main\etc\conda\activate.d\env_vars.bat
```

系统地最新版本的cuda为新安装的，有两个重复的cuda，可能与path路径排序有关，在power shell中，尽管nvcc -V输出的是11.3的 cuda 版本，可以在编译的时候还是出现 detected 和 compile 不一致的关系，emmm，而且对于 setup 文件中出现的 cuda home 也同样为 10.2 的路径emmm，直接改变 home path并不起作用似乎是

是的，直接在python 文件改变 home path 并不能够起作用，需要修改 用户变量 path，这里的修改而且很有讲究，应该是要修改， CUDA_PATH，而且直接修改也不行，还需要重启电脑emmm，不敢想在服务器是怎么办法emmm

python setup.py clean

![image-20240922164604504](C:\Users\liujialong\AppData\Roaming\Typora\typora-user-images\image-20240922164604504.png)

```
The detected CUDA version (10.2) mismatches the version that was used to compile
PyTorch (11.3). Please make sure to use the same CUDA versions.
```

6、在1.11.0 pytorch+cuda11.3 版本和 系统 11.3 cuda版本进行编译，python3.8 出现报错

```
4\FOCS\FCOS-master01\FCOS-master\build\temp.win-amd64-cpython-38\Release\fcos_core\csrc\cpu\_C.cp38-win_amd64.lib
LINK : fatal error LNK1181: 无法打开输入文件“C:\Ob_De\2024\FOCS\FCOS-master01\FCOS-master\build\temp.win-amd64-cpython-38\Release\fcos_core\csrc\cpu\ROIAlign_cpu.obj”
error: command 'C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.29.30133\\bin\\HostX86\\x64\\link.exe' failed with exit code 1181
```

后续这个错误无法继续跟踪，源码没有做任何改变，修改了另一个env 环境 是python39，可能还有别的一些包也不一样，就没有再出现上述的链接出问题了，知道为什么没有出问题了，因为该包没有use_ninja，当将该env 装上ninja的时候还是会出现上述链接问题emmm，可能是版本不对，装上最新的ninja还是会有问题emmm，同样的无法链接文件问题

​	该代码出现的问题很可能是 cuda11.3 版本过高导致的问题，因为之前时候 cuda102的时候没有出现类似的问题emmm

是的，重装一个env后，3.7 python conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch环境下，如果没有ninja就不会出现链接问题，一旦安装了 ninja，会出现链接obj文件问题，而且是新问题，不仅和ninja有关，也应该和pytorch版本，cuda 版本有关

```
LINK : fatal error LNK1181: 无法打开输入文件“C:\Ob_De\2024\FOCS\FCOS-master01\FCOS-master\build\temp.win-amd64-cpython-37\Release\fcos_core\csrc\cuda\deform_conv_cuda.obj”
error: command 'C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.29.30133\\bin\\HostX86\\x64\\link.exe' failed with exit code 1181
```



```
. 缺少目标文件
Ninja 尝试链接某个 .obj 文件或其他中间文件（如 .lib、.dll）时，找不到该文件。常见原因是编译失败，导致某些中间文件未生成。
解决方法：

检查所有需要生成的目标文件（如 .obj 文件）是否在之前的编译步骤中成功生成。
通过 Ninja 日志，确定哪些文件没有生成，并查找编译过程中是否有报错。
可以通过以下命令手动运行 Ninja 并检查输出日志：

bash
复制代码
ninja -v
2. 路径问题
文件路径可能包含特殊字符、空格或长度超过系统的路径长度限制，导致文件无法找到。
解决方法：

确保输入文件的路径没有特殊字符，路径名不要过长。
检查构建系统是否使用了正确的相对路径或绝对路径。可在 setup.py 或 CMakeLists.txt 中检查路径配置。
3. 中间文件被删除
某些编译生成的中间文件（如 .obj 文件）可能在编译后被意外删除或被清理工具清理掉，导致 Ninja 无法找到这些文件进行链接。
解决方法：

在清理构建目录后，重新运行编译：
bash
复制代码
python setup.py clean
python setup.py build_ext
4. 并行编译冲突
Ninja 默认进行并行编译，有时多个编译任务同时进行会导致某些文件被遗漏，或者出现依赖未完全生成的情况。
解决方法：

降低并行编译的并行度或禁用并行编译，使用单线程运行编译：
bash
复制代码
python setup.py build_ext --jobs=1
这可以避免并行构建时的依赖冲突。
5. 链接器问题
在某些情况下，Ninja 的链接器无法找到 .obj 文件或 .lib 文件的原因是链接器未正确配置或者需要的库文件未正确指定。
解决方法：

检查 setup.py 或 CMakeLists.txt 中的链接配置，确保所有需要链接的库文件（如 .lib 或 .dll 文件）都存在并正确指定。
6. 编译环境问题
编译环境不正确（例如 Visual Studio 版本不兼容、CUDA 工具链不正确等）也可能导致文件找不到。
解决方法：

确认你正在使用正确的编译环境。
检查 Visual Studio 的 vcvarsall.bat 是否已正确加载编译环境，确保它可以找到所需的工具链。
具体操作步骤：
查看 Ninja 输出日志：

bash
复制代码
ninja -v
检查是否有文件编译失败，或者 Ninja 正在寻找哪些特定的文件。

清理构建缓存并重新构建：

bash
复制代码
python setup.py clean
python setup.py build_ext
检查是否生成了 .obj 文件：

确保 ninja 或 setup.py 指定的路径中确实存在生成的 .obj 文件（例如 ROIAlign_cpu.obj）。
确认所有目标文件已经成功生成。
通过这些步骤，应该可以帮助你定位并解决 LNK1181 错误。
```

7、PyCharm Community Edition 2023.1.1 有时候也犯混，对于一个简单的pyhon 文件竟然不能够调试，奇了怪了，但是只有打断点打的足够早，而且也要足够多，在import 附近，会发现能够调试了emmm

8、还有关于库文件的问题

```
ROIAlign_cuda.cu(5): fatal error C1083: 无法打开包括文件: “THC/THC.h”: No such file or directory
```

原因是maskrcnn编译时报错，原因是我装的torch是1.13的，但THC.h文件在Pytorch1.10版之后被移除了。

这是pytorch 版本的问题emmm，可以选择换版本，也可以选择换头文件，换头文件的话还是很复杂的，因为涉及到许多的文件

```
将文件中的#include <THC/THC.h>注释掉
THCudaCheck函数报错。将所有该函数替换为AT_CUDA_CHECK
THCCeilDiv函数报错。在文件头部添加该函数：
```

9、还有一些warning

```
UserWarning: The detected CUDA version (11.3) has a minor version mismatch with the version that was used to compile PyTorch (11.8). Most likely this shouldn't be a problem.
```



10、重新复现版本 conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

python 3.7 这次 conda env 创建十分快，应该是有备份， 然后按照惯例， 创建 activate.d  文件夹和 env_vars.bat 文件，

然后重新激活环境，就可以编译了，但是还是会setup.py 文件的 CUDA_HOME 仍然是11.3，但并不会出现 detected 和 compile pytorch 不匹配的问题，报错

```
fcos_core/csrc/cuda/deform_conv_cuda.cu(586): error: identifier "AT_CHECK" is undefined
```

这个问题也是torch版本过高的原因

```
error: ‘AT_CHECK’ was not declared in this scope
在编译deform_conv时遇到问题：error: ‘AT_CHECK’ was not declared in this scope

错误原因：AT_CHECK is deprecated in torch 1.5
高版本的pytorch不再使用AT_CHECK，而是使用 TORCH_CHECK。

解决方法：将所有待编译的源文件中的‘AT_CHECK’全部替换为‘TORCH_CHECK’
```

​	还有个问题要记录，就是禁用ninja，不建议禁用，因为最后成功的版本并未禁用，而且禁用后要修改一大堆头文件，最后也不一定有效emmm，修改头文件的话emmm，后续可以尝试复现 cu10::Half的问题emmm

```
原因：

通过使用ninja，对xxx.cpp文件进行编译 ，生成xxx.obj（linux中是生成xxx.o）失败，使用ninja无法生成xxx.obj，导致出错

解决办法：

将ninja关闭，修改如图所示，添加红框内的内容
```

然后禁用ninja, 然后将部分文件的 AT_CHECK 替换为 TORCH_CHECK 即可， 不仅要运行 python setup.py build_ext 

还需要 python setup.py develop， 应该就OK了，该过程之前可能要装一部分python库

这时候运行会迟迟不出结果，调试后发现还有一个warning，并且运行卡在了

```
C:\Users\liujialong\.conda\envs\FCOS_A\lib\site-packages\torch\cuda\__init__.py:143: UserWarning: 
NVIDIA GeForce RTX 3060 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 compute_37.
If you want to use the NVIDIA GeForce RTX 3060 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
```

查找原因是因为，应该是 cuda 版本较低导致的问题，从而无法跑通代码，并且也不报错emmm

```
最常见的解决方式是升级Pytorch版本，新的版本增加了对新显卡架构的支持。但是有时候升级到1.10.0问题仍然没有解决，其实1.7.1版本的pytorch就已经支持3090，问题没有解决的原因大概率是CUDA版本的问题。3090显卡一般使用CUDA11+，而直接pip安装的pytorch可能是cuda10.2版本的，所以只依靠升级pytorch版本是不行的，还需要安装对应cuda版本的pytorch。
```



11、接下来我们启用较高版本的 cuda 为 11.3，对应pytorch 1.11.0 的 cuda 编译版本也是 11.3，中途应该清理以下build 文件夹下存在的代码，否则遗留的一些编译文件会导致即使编译通过了，最后也会报错的问题, 这里应该禁用ninja编译，因为会导致编译失败，即使 set maxjobs=1，也会报一个编译错误；前面已经解释过了，和包的版本相关联，而可能并非源码问题

```
然后在 python.3.8 以及 pytorch1.11.0, torch.version.cuda '11.3', 和 nvcc  -V 
Built on Mon_May__3_19:41:42_Pacific_Daylight_Time_2021
Cuda compilation tools, release 11.3, V11.3.109
Build cuda_11.3.r11.3/compiler.29920130_0
```

的环境下试验，首先会报错，原因是THC.h文件在Pytorch1.10版之后被移除了

```
fatal error C1083: 无法打开包括文件: “THC/THC.h”: No such file or directory
```

然后尝试：

```
可以看到，在/maskrcnn_benchmark/csrc/cuda文件夹中的所有以.cu结尾的代码文件中删除了下述头文件：
#include <THC/THC.h>
并且把所有的
THCudaCheck(cudaGetLastError());
替换成了
AT_CUDA_CHECK(cudaGetLastError());
```

还有 error: identifier "THCCeilDiv" is undefined， 也是 THC 头的问题v

```
现 "identifier THCCeilDiv is undefined" 错误，通常是由于 PyTorch 的 API 或底层代码在不同版本之间发生变化，导致某些标识符在新版本中被移除、替换或重构。THCCeilDiv 这个符号曾经出现在早期的 PyTorch 和 CUDA 代码中，但在最新的版本中可能已被移除或替代。
```

解决办法是：

```
引入头文件<ATen/ceil_div.h>，然后用at::ceil_div来替换THCCeilDiv ，
比如 int blocks = THCCeilDiv(total_elements, threads_per_block);
替换为
#include <ATen/ceil_div.h>
// 使用at::ceil_div替代
int blocks = at::ceil_div(total_elements, threads_per_block);
```

还有一个问题是，有关于一个atomicAdd的问题，尝试了很久很久，没能够解决它，包括写一个替换的函数，包括强制转换类型，均无法成功，还有重载函数之类的，

```
C:\Ob_De\2024\FOCS\FCOS-master01\FCOS-master\fcos_core\csrc\cuda\deform_conv_kernel_cuda.cu(695): error: no instance of overloaded function "atomicAdd" matches the argument list
            argument types are: (c10::Half *, c10::Half)
          detected during instantiation of "void modulated_deformable_col2im_gpu_kernel(int, const scalar_t *, const scalar_t *, const scalar_t *, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, scalar_t *) [with scalar_t=c10::Half]" 
(820): here
```

问题的原因是，ChatGPT回答，但是似乎是显卡算例的问题，应该是 AtomicAdd 只支持比较老的 Cuda 版本，因为那个时候没有 Half 类型的参数传入，而现在使用比较新的 Cuda 版本编译，传入了 Half 类型的参数? 对于 scalr_t， 应该是这样的，这样就解释得通畅了，

还有一种解法好像是：***Turns out I should use gpuAtomicAdd rather than atomicAdd. Replacing solved the problem! Thanks!***

```
atomicAdd 不支持 c10::Half 类型: CUDA 的 atomicAdd 函数原生只支持 32位浮点数 (float) 和 64位浮点数 (double)。然而，c10::Half 是 16位浮点数，也叫 half-precision，CUDA 默认不支持对 half 类型进行 atomicAdd 操作。

PyTorch 与 CUDA 版本的兼容性问题: PyTorch 在其较新的版本中对 half-precision 的支持有所增强，特别是在 TensorCore 加速器存在的 GPU（如 Volta、Turing、Ampere 等架构的 GPU）上。这个错误可能发生在使用不支持的 atomicAdd 操作时，特别是当你使用 PyTorch 的 c10::Half 类型来表示半精度浮点数，且你正在进行可能需要原子操作的并行计算时。

modulated_deformable_col2im_gpu_kernel 内核使用 atomicAdd: 在该内核中，可能正在对某些张量元素进行并发累加（即多个线程可能需要对同一位置进行累加），而累加操作需要是原子的以避免竞争条件。但由于你使用的是 c10::Half 类型，这触发了 atomicAdd 不支持 c10::Half 的问题。
```

```
CUDA Driver Version / Runtime Version          12.3 / 11.3
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 12288 MBytes (12884377600 bytes)
  (28) Multiprocessors, (128) CUDA Cores/MP:     3584 CUDA Cores
  GPU Max Clock rate:                            1867 MHz (1.87 GHz)
  Memory Clock rate:                             7501 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 2359296 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               zu bytes
  Total amount of shared memory per block:       zu bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          zu bytes
  Texture alignment:                             zu bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.3, CUDA Runtime Version = 11.3, NumDevs = 1, Device0 = NVIDIA GeForce RTX 3060
Result = PASS
```

对于c10:Half 问题

使用如下头文件去替换 THCAtomics.cuh 头文件emmm，有一些效果

```
#pragma once
// TODO: Remove once torchvision has been updated to use the ATen header
#include <ATen/cuda/Atomic.cuh>
```

随后我查看了头文件，部分 THCAtomics.cuh的问题件内容直接 改为 #include <ATen/cuda/Atomic.cuh>

还有一部分THCAtomics.cuh的头文件中有Half定义,既然有定义，为什么还会报错呢？

```
static inline __device__ at::Half atomicAdd(at::Half *address, at::Half val) {
  return gpuAtomicAdd(address, val);
}
```

chatgpt 给出的回答是：

```
CUDA 版本问题 ： 如果你使用的 CUDA 版本不支持 `__half` 的 `atomicAdd`（CUDA 7.5 或更低版本），需要手动实现 `atomicAdd`，正如你在代码中已经做的那样。检查你的 CUDA 版本，确保它是 CUDA 8.0 或更高版本，因为从 CUDA 8.0 开始，`atomicAdd` 才支持 `__half`。

**不匹配的数据类型**： 你在定义中使用的是 `at::Half`，而错误信息中提到的类型是 `c10::Half`。虽然 `at::Half` 和 `c10::Half` 是等价的（`at::Half` 是 `c10::Half` 的别名），但你应该确保在所有地方使用一致的类型。

- 尝试在 `atomicAdd` 函数定义中将 `at::Half` 改为 `c10::Half`。
```

但是很明显，这里的Half实现借助了：

```
static inline  __device__ at::Half gpuAtomicAdd(at::Half *address, at::Half val) {
#if defined(USE_ROCM) || ((defined(CUDA_VERSION) && CUDA_VERSION < 10000) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  return AtomicFPOp<at::Half>()(address, val,
                                [](at::Half hsum, at::Half val) {
                                  return hsum + val;
                                });
#else
  return atomicAdd(reinterpret_cast<__half*>(address), val);
#endif
}
```

然后观察：，这里给出的一种解释是如果CUDA 的版本较低，就采用 AtomicFPOp 的实现方式，而如果 CUDA 版本较高，则采用 atomicAdd 的实现方式，形成一种类似套娃的机制，因此，在CUDA 较高版本中，无法仅仅通过 头文件 THCAtomics.cuh 去 支持，而应该改变头文件，这是一种可能的解释方式。

也有可能是pytorch版本和cuda版本共同作用的结果

随之而来有三个问题，分别是

```
error: identifier "THCudaFree" is undefined error: identifier "THCudaMalloc" is undefined error: identifier "state" is undefined怎么解决啊
```

这个问题的解决方法是：

```
旧代码：
THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState
mask_dev = (unsigned long long*) c10::cuda::CUDACachingAllocator::raw_alloc(state, boxes_num * col_blocks * sizeof(unsigned long long));
c10::cuda::CUDACachingAllocator::raw_delete(state, mask_dev);
新代码：
// 无需 THCState，直接使用 CUDACachingAllocator
mask_dev = (unsigned long long*) c10::cuda::CUDACachingAllocator::raw_alloc(boxes_num * col_blocks * sizeof(unsigned long long));
// 执行操作后，释放内存
c10::cuda::CUDACachingAllocator::raw_delete(mask_dev);
```

接着继续报错：

```
报错error: name followed by "::" must be a class or namespace name
在mask_dev = (unsigned long long*) c10::cuda::CUDACachingAllocator::raw_alloc(boxes_num * col_blocks * sizeof(unsigned long long));是什么原因
```

这里应该是缺少了头文件

```
#include <c10/cuda/CUDACachingAllocator.h>
```

之后还有一部分问题，都是重复的问题，然后就可以在 如下的环境运行FCOS 代码了，而且不会报错，能够正常收敛

```
3060
cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False)},
torch.version.cuda '11.3'
nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Mon_May__3_19:41:42_Pacific_Daylight_Time_2021
Cuda compilation tools, release 11.3, V11.3.109
Build cuda_11.3.r11.3/compiler.29920130_0
pytorch对应的版本1.11.0, cuda 编译版本同 torch.version.cuda '11.3'
python版本为 3.8.19
改完源码后，还是不能开启ninja编译工具会报错
无法打开输入文件“C:\Ob_De\2024\FOCS\FCOS-master01\FCOS-master\build\temp.win-amd64-cpython-38\Release\fcos_core\csrc\cpu\ROIAlign_cpu.obj”

```

12、从上面的内容我们可以推导出，pytorch版本不应该高于 1.10.0，否则会导致缺少很多头文件，而Cuda的版本因该要支持 sm86，不应该低于10.2的版本

然后我恰好安装了 

```
python=3.7.16
pytorch=1.10.0
cuda 编译 10.2
pytorch-cuda=10.2
编译设置 cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False)},
编译设置 cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},还是不行
只需要改动AT_CHECK即可
编译能够通过，也不会报错，但是代码就是不能训练，可以运行
NVIDIA GeForce RTX 3060 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 compute_37.
If you want to use the NVIDIA GeForce RTX 3060 GPU with PyTorch, please check the instructions at
```

会卡在

```
2024-09-22 22:54:29,164 fcos_core.trainer INFO: Start training
```

就是卷积运算卡住问题

现在我们将pytorch版本得到pytorch1.10.0；

现在一种可能的配置如下

```
python=3.7.16
pytorch=1.10.0
cuda 编译 11.3
pytorch-cuda=11.1
编译设置 cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False)},
编译设置 cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},还是不行
可以运行，只需要改动少量代码
```

13、

总结，因此对于 cuda 版本过低的问题，导致一些精度无法适配，例如 Half 精度；这个Half精度也可能是 torch版本过高导致的，而且cuda 版本过低，sm86不支持

而对于pytorch版本过高的问题，很多头 cuda 文件都不适配，因为pytorch本身和cuda是一起编译出来的内容；cuda10.2始终没有成功，但是也学到了很多东西

贴一个最早出现的安装配置

```
python=3.7.16
pytorch=1.7.1
cuda 编译 11.3
pytorch-cuda=11.0
编译设置 cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False)},
编译设置 cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},还是不行
可以运行，只需要改动少量代码
```

```
较高pytorch版本的 3060
cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False)},
torch.version.cuda '11.3'
Build cuda_11.3.r11.3/compiler.29920130_0
pytorch对应的版本1.11.0, cuda 编译版本同 torch.version.cuda '11.3'
python版本为 3.8.19
但是要改动极其大量的代码
```

14、一般就是环境问题，emmm太难受了，不过还好，两天总算跑通了一份代码，FCOS成了，ATSS代码也就成了，学习以下emmm，明天看看能不能出个基线，周六遇到的问题周日都复现了，下一步学一下这个 python build 到底是什么东西emmm
