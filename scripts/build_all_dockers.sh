#!/bin/bash
set -xe
script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build env for Souffle, XLA and Apollo
cd ${script_directory}/.. && bash run_tvm_0.8.sh build
# Build env for TensorRT
cd ${script_directory}/../baseline/tensorrt && bash build_docker_tensorrt.sh
# Build env for Rammer
cd ${script_directory}/../baseline/rammer && bash build_docker_rammer.sh
# Build env for IREE
cd ${script_directory}/../baseline/iree && bash run_docker_iree.sh build
