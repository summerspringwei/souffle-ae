#!/bin/bash
set -xe
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

# Run the efficientnet_se_module_unittest
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} /bin/bash -c "/workspace/souffle-models/cpp/efficientnet/run_efficient_se_module_unittest.sh"
docker cp ${souffle_container_id}:/workspace/souffle-models/cpp/efficientnet/scripts/efficientnet-se-module-latency-ours.pdf ${script_directory}/../results/
docker cp ${souffle_container_id}:/workspace/souffle-models/cpp/efficientnet/scripts/efficientnet-se-module-latency-ours.eps ${script_directory}/../results/
docker cp ${souffle_container_id}:/workspace/souffle-models/cpp/efficientnet/scripts/efficientnet-se-module-latency-ours.png ${script_directory}/../results/
