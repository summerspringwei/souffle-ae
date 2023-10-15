ARG CUDA_VERSION=11.7.1
ARG OS_VERSION=18.04

# FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${OS_VERSION}
FROM nvidia/cuda:11.7.1-devel-ubuntu18.04

ENV TZ=Asia/Kolkata \
    DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN mkdir -p /workspace
WORKDIR /workspace

# Install basic dependencies.
RUN apt-get update && apt-get install \
  -y wget vim git python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz \
    && tar -xf clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz \
    && rm clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz \
    && mv clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04 clang_10.0.0 \
    && ln -s /clang_10.0.0/bin/clang /usr/bin/clang \
    && ln -s /clang_10.0.0/bin/clang++ /usr/bin/clang++ \
    && ln -s /clang_10.0.0/bin/llvm-config /usr/bin/llvm-config

# Install nsight-compute and nsight-systems
RUN apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         apt-transport-https \
         ca-certificates \
         gnupg \
         wget && \
     rm -rf /var/lib/apt/lists/*
RUN  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
     wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
         apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         nsight-compute-2022.4.1 cuda-nsight-systems-11-7 && \
     rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/nvidia/nsight-compute/2022.4.1/:${PATH}


# Install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh \
    && bash Anaconda3-2022.05-Linux-x86_64.sh -b -p /workspace/anaconda3 \
    && rm Anaconda3-2022.05-Linux-x86_64.sh \
    && echo "export PATH=/workspace/anaconda3/bin:$PATH" >> ~/.bashrc

#  Build and install TVM
RUN git clone --recursive https://github.com/apache/tvm /workspace/tvm \
    && cd /workspace/tvm \
    && git checkout v0.8 \
    && git submodule init \
    && git submodule update \
    && mkdir build \
    && mkdir /workspace/tvm/dbg_build \
    && cp cmake/config.cmake build 
COPY config.cmake /workspace/tvm/build/config.cmake
COPY patch_module_bench_tvm_0.8.patch /workspace/tvm/patch_module_bench_tvm_0.8.patch
RUN cd /workspace/tvm && git apply patch_module_bench_tvm_0.8.patch

# Build release and debug version tvm
RUN cd /workspace/tvm/build && cmake .. \
    && make -j20 \
    && cd /workspace/tvm/dbg_build \
    && cmake -DCMAKE_BUILD_TYPE=Debug .. \
    && make -j20

# Install xgboost for auto_scheduler
RUN /workspace/anaconda3/bin/pip install xgboost==1.5.0 
RUN /workspace/anaconda3/bin/pip install numpy==1.21.0
RUN /workspace/anaconda3/bin/pip install torch==2.0.1 torchvision torchaudio


RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8-dev_8.5.0.96-1+cuda11.7_amd64.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8_8.5.0.96-1+cuda11.7_amd64.deb && \
    dpkg -i libcudnn8_8.5.0.96-1+cuda11.7_amd64.deb && \
    dpkg -i libcudnn8-dev_8.5.0.96-1+cuda11.7_amd64.deb

RUN apt list --installed | grep cudnn

RUN apt update -y && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends gdb

RUN mkdir /workspace/third_party && cd /workspace/third_party && git clone https://github.com/lukemelas/EfficientNet-PyTorch.git
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libcudnn8

RUN cd /workspace && git clone https://github.com/abseil/abseil-cpp.git && mkdir abseil-cpp/build && cd abseil-cpp/build && cmake .. && make -j20 && make install

# RUN cd to_python_binding_path && python setup.py clean && python setup.py install && pip install .

RUN /workspace/anaconda3/bin/pip install tensorflow==2.10.0 yacs tensorflow-addons 


# Build mindspore according to https://gitee.com/mindspore/docs/blob/r1.3/install/mindspore_gpu_install_source.md
RUN apt update -y && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends libnuma-dev

RUN cd /workspace && git clone https://gitee.com/mindspore/mindspore.git -b r1.3 

COPY souffle-mindspore-patch.txt /workspace/mindspore

RUN apt update -y && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends libgmp-dev
ENV PATH=/workspace/anaconda3/bin:$PATH
RUN cd /workspace/mindspore && git apply souffle-mindspore-patch.txt 

RUN apt update -y && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends patch automake libtool flex
ENV CC="/usr/bin/gcc"
ENV CXX="/usr/bin/g++"
RUN cd /workspace/mindspore && bash build.sh -e gpu
RUN cd /workspace/mindspore && /workspace/anaconda3/bin/pip install output/mindspore_gpu-1.3.0-cp39-cp39-linux_x86_64.whl
# mindspore requires CUDA 11.1
# RUN apt update -y && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends cuda-11-1
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-11-1_11.1.1-1_amd64.deb && \
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-runtime-11-1_11.1.1-1_amd64.deb && \
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-toolkit-11-1_11.1.1-1_amd64.deb && \
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-demo-suite-11-1_11.1.74-1_amd64.deb \
#     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-libraries-11-1_11.1.1-1_amd64.deb

# RUN dpkg -i cuda-runtime-11-1_11.1.1-1_amd64.deb && \
#     dpkg -i cuda-toolkit-11-1_11.1.1-1_amd64.deb && \ 
#     dpkg -i cuda-demo-suite-11-1_11.1.74-1_amd64.deb && \
#     dpkg -i cuda-11-1_11.1.1-1_amd64.deb

RUN wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
RUN bash cuda_11.1.0_455.23.05_linux.run --silent --toolkit
RUN rm /usr/local/cuda
RUN ln -s /usr/local/cuda-11.7 /usr/local/cuda

RUN git clone https://gitee.com/mindspore/models.git /workspace/mindspore_models && \
    cd /workspace/mindspore_models && git checkout daae6cd7d72ba0912209924bd5d7b8345d31c4ee
COPY souffle-mindspore_models-patch.txt /workspace/mindspore_models/
RUN cd /workspace/mindspore_models/ && git apply souffle-mindspore_models-patch.txt
RUN mkdir /workspace/baseline

# Set and modify environment variables here
ENV PYTHONPATH=/workspace/tvm/python:/workspace/third_party/EfficientNet-PyTorch:${PYTHONPATH}
ENV PYTHONPATH=/workspace/souffle-models/python/:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/workspace/tvm/build:${LD_LIBRARY_PATH}
ENV TORCH_HOME="/workspace/anaconda3/lib/python3.9/site-packages/torch"
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${TORCH_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib/stubs:${CUDA_HOME}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
ENV CUTLASS_HOME="/workspace/cutlass"
RUN echo "Build docker!"
# For mindspore
# ENV CUDA_HOME_11=/usr/local/cuda-11.1
# ENV LD_LIBRARY_PATH=${CUDA_HOME_11}/lib64:${CUDA_HOME_11}/targets/x86_64-linux/lib/stubs:${CUDA_HOME_11}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
# RUN python -c "import mindspore;mindspore.run_check()"
