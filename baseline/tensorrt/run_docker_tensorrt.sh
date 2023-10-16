#!/bin/bash
set -xe

cd $(pwd)/TensorRT && ./docker/launch.sh --tag souffle-tensorrt8.4.1-ubuntu18.04 --gpus all