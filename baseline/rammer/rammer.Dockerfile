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

RUN cd /workspace && wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run

RUN bash cuda_10.2.89_440.33.01_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-10.2 --override

ENV CUDA_HOME=/usr/local/cuda-10.2
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib/stubs:${CUDA_HOME}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH