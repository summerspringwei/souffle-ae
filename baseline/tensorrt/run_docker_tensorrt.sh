#!/bin/bash
set -xe
script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${script_directory}/TensorRT && ./docker/launch.sh --tag souffle-tensorrt8.4.1-ubuntu18.04 --gpus all
