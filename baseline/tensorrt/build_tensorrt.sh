set -xe
# git clone -b main https://github.com/nvidia/TensorRT TensorRT
# cd TensorRT
# git submodule update --init --recursive
# git checkout 8.4.1
# # We need to checkout the corresponding version onnx parse
# cd parsers/onnx/ && git checkout release/8.4-GA && git submodule sync && git submodule update --init --recursive && cd ../../
# Build TensorRT
cd TensorRT && ./docker/build.sh --file docker/ubuntu-18.04.Dockerfile --tag souffle-tensorrt8.4.1-ubuntu18.04
# Launch the docker container
./docker/launch.sh --tag souffle-tensorrt8.4.1-ubuntu18.04 --gpus all
# Build TensorRT
trt_container_id=$(docker ps -qf "ancestor=souffle-tensorrt8.4.1-ubuntu18.04:latest")
docker exec -it ${trt_container_id} \
  /bin/bash -c "cd /workspace/TensorRT && mkdir build && cd build && cmake .. && make -j$(nproc) && sudo make install"
# MOUNT_MODEL_FOLDER="-v /home2/xiachunwei/Software/fusion/tensorrt-8.4-engines/:/workspace/tensorrt-8.4-engines/"