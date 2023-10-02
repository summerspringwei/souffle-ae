## Prepare for IREE Env

IREE requires latest python3.11.
First download latest anaconda:
```shell
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
bash Anaconda3-2023.07-2-Linux-x86_64.sh -b -p ${HOME}/anaconda3.11.4
source ${HOME}/anaconda3.11.4/bin/activate
```
Create virtual env and install iree
```shell
python3.11 -m venv mlir_venv
source mlir_venv/bin/activate
pip install https://github.com/llvm/torch-mlir/releases/download/oneshot-20230816.122/torch-2.1.0.dev20230816+cpu-cp311-cp311-linux_x86_64.whl
pip install https://github.com/llvm/torch-mlir/releases/download/oneshot-20230816.122/torch_mlir-20230816.122-cp311-cp311-linux_x86_64.whl
pip install https://download.pytorch.org/whl/nightly/cpu/torchvision-0.16.0.dev20230816%2Bcpu-cp311-cp311-linux_x86_64.whl
pip install Pillow
pip install requests
```

Install iree
```shell
pip install https://github.com/openxla/iree/releases/download/candidate-20230816.17/iree_compiler-20230816.17-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install https://github.com/openxla/iree/releases/download/candidate-20230816.17/iree_runtime-20230816.17-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install https://github.com/openxla/iree/releases/download/candidate-20230816.17/iree_tools_tf-20230816.17-py3-none-any.whl
```
Convert torch module to mlir
```
python3 convert_efficient_to_mlir.py
```

Using iree to compile and run:
```shell
iree-compile --iree-hal-target-backends=vmvx ./efficientnet-b0.mlir -o efficientnet-b0.cpu.vmfb
iree-run-module --module=efficientnet-b0.cpu.vmfb --input="1x3x224x224xf32=@dummy_input.npy"
```

All env are on ACT102 `~/Software/mlir_venv`
Compile mlir to cuda vmfb and run
```shell
iree-compile --iree-hal-target-backends=cuda efficientnet-b0.mlir -o efficientnet-b0.cuda.vmfb
iree-run-module --device=cuda --module=efficientnet-b0.cuda.vmfb --input="1x3x224x224xf32@dummy_input.npy"
```


## Convert onnx to mlir
```python
import torch
import torch_mlir
import io
import iree.compiler as ireec

model = # ... the model we're compiling
example_input = # ... an input to the model with the expected shape and dtype
mlir = torch_mlir.compile(
    model,
    example_input,
    output_type="linalg-on-tensors",
    use_tracing=True)

iree_backend = "llvm-cpu"
iree_input_type = "tm_tensor"
bytecode_stream = io.BytesIO()
mlir.operation.write_bytecode(bytecode_stream)
iree_vmfb = ireec.compile_str(bytecode_stream.getvalue(),
                              target_backends=[iree_backend],
                              input_type=iree_input_type)


```
