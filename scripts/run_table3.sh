#!/bin/bash
set -x
script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SOUFFLE_RUN=$1

if [ -e "${script_directory}/../results/table3.csv" ]; then
  rm -f ${script_directory}/../results/table3.csv
fi
touch ${script_directory}/../results/table3.csv

# First launch the docker container
if [ ! "$(docker ps -qf "ancestor=souffle-tvm-0.8:latest")" ]; then
  bash ../run_tvm_0.8.sh run
fi
souffle_container_id=$(docker ps -qf "ancestor=souffle-tvm-0.8:latest")
echo ${souffle_container_id}


# XLA Pass
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} /bin/bash -c "/workspace/baseline/xla/run_xla.sh"
docker exec -it ${souffle_container_id} /bin/bash -c "cat /workspace/baseline/xla/xla_models/table3_xla.csv" >> ${script_directory}/../results/table3.csv

# Require docker
# TensorRT Pass
if [ ! "$(docker ps -qf "ancestor=souffle-tensorrt8.4.1-ubuntu18.04:latest")" ]; then
  bash ../baseline/tensorrt/run_docker_tensorrt.sh
fi
trt_container_id=$(docker ps -qf "ancestor=souffle-tensorrt8.4.1-ubuntu18.04:latest")
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${trt_container_id} /bin/bash /workspace/tensorrt-8.4-engines/run_tensorrt_models.sh
docker exec -it ${trt_container_id} \
  /bin/bash -c "cat /workspace/tensorrt-8.4-engines/table3_tensorrt.csv" >> ${script_directory}/../results/table3.csv

# Require docker
# Rammer Pass
if [ ! "$(docker ps -qf "ancestor=sunqianqi/sirius:mlsys_ae")" ]; then
bash ${script_directory}/../baseline/rammer/run_docker_rammer.sh start
fi
bash ${script_directory}/../baseline/rammer/run_docker_rammer.sh run
cat ${script_directory}/../baseline/rammer/rammer_generated_models/table3_rammer.csv >> ${script_directory}/../results/table3.csv


# Apollo Pass
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} /bin/bash /workspace/baseline/mindspore/run_nsys_apollo.sh
docker exec -it ${souffle_container_id} \
  /bin/bash -c "cat /workspace/baseline/mindspore/table3_apollo.csv" >> ${script_directory}/../results/table3.csv

# IREE Pass
if [ ! "$(docker ps -qf "ancestor=souffle-iree:latest")" ]; then
  bash ../baseline/iree/run_docker_iree.sh run
fi
iree_container_id=$(docker ps -qf "ancestor=souffle-iree:latest")
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${iree_container_id} /bin/bash /workspace/iree_models/run_nsys_iree.sh
docker exec -it ${iree_container_id} \
  /bin/bash -c "cat /workspace/iree_models/table3_iree.csv" >> ${script_directory}/../results/table3.csv

# Ours Pass
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} /bin/bash /workspace/souffle-models/python/models/run_ncu_souffle.sh
docker exec -it ${souffle_container_id} \
  /bin/bash -c "cat /workspace/souffle-models/python/models/table3_souffle.csv" >> ${script_directory}/../results/table3.csv
