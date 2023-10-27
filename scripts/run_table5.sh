#!/bin/bash

# Run TensorRT
set -x
script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SOUFFLE_RUN=$1
table5_path=${script_directory}/../results/table5.csv

if [ -e "${table5_path}" ]; then
  rm -f ${table5_path}
fi
touch ${table5_path}

# First launch the docker container if container is not run
if [ ! "$(docker ps -qf "ancestor=souffle-tvm-0.8:latest")" ]; then
  bash ../run_tvm_0.8.sh run
fi
souffle_container_id=$(docker ps -qf "ancestor=souffle-tvm-0.8:latest")
echo "Souffle docker running: " ${souffle_container_id}

# Launch TensorRT docker
if [ ! "$(docker ps -qf "ancestor=souffle-tensorrt8.4.1-ubuntu18.04:latest")" ]; then
  bash ${script_directory}/../baseline/tensorrt/run_docker_tensorrt.sh
fi
tensorrt_container_id=$(docker ps -qf "ancestor=souffle-tensorrt8.4.1-ubuntu18.04:latest")
echo "TensorRT docker running": 

# Run ncu to get the kernel number and memory read
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${tensorrt_container_id} /bin/bash -c "cd /workspace/tensorrt-8.4-engines && ./run_ncu_tensorrt.sh"
# Cat result to table5.csv
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${tensorrt_container_id} /bin/bash -c "cat /workspace/tensorrt-8.4-engines/table5_tensorrt.csv" >> ${table5_path}

# Run Apollo
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} /bin/bash -c "cd /workspace/baseline/mindspore && ./run_ncu_apollo.sh"
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} /bin/bash -c "cat /workspace/baseline/mindspore/table5_apollo.csv" >> ${table5_path}

# Run XLA
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} /bin/bash -c "cd /workspace/baseline/xla && ./run_ncu_xla.sh"
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} /bin/bash -c "cat /workspace/baseline/xla/xla_models/table5_xla.csv" >> ${table5_path}

# Run souffle
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} /bin/bash -c "cd /workspace/souffle-models/python/models && ./run_ncu_mem_souffle.sh"
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} /bin/bash -c "cd /workspace/souffle-models/python/models && cat table5_souffle.csv" >> ${table5_path}
