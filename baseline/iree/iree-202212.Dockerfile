FROM nvcr.io/nvidia/pytorch:22.05-py3
RUN git clone https://github.com/openxla/iree.git
RUN cd iree && git checkout candidate-20221230.375 && git submodule update --init && cmake -DIREE_TARGET_BACKEND_CUDA=ON -DIREE_HAL_DRIVER_CUDA=ON -G Ninja -B ../iree-build/ .
WORKDIR /workspace
RUN cmake --build iree-build/
