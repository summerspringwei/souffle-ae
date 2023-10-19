#!/bin/bash
set -xe
script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SOUFFLE_RUN=$1

if [ -e "${script_directory}/../results/table4.csv" ]; then
  rm -f ${script_directory}/../results/table4.csv
fi
touch ${script_directory}/../results/table4.csv

# First launch the docker container
if [ ! "$(docker ps -qf "ancestor=souffle-tvm-0.8:latest")" ]; then
  bash ../run_tvm_0.8.sh run
fi
souffle_container_id=$(docker ps -qf "ancestor=souffle-tvm-0.8:latest")
echo ${souffle_container_id}
souffle_model_path="/workspace/souffle-models/python/models/"

# BERT
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} \ 
  /bin/bash -c "/workspace/souffle-models/python/models/bert/run_ncu_bert.sh"
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} \ 
  /bin/bash -c "cat ${souffle_model_path}/bert/table4_bert.csv >> ${script_directory}/../results/table4.csv"

# ResNext
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} \ 
  /bin/bash -c "/workspace/souffle-models/python/models/resnext/run_ncu_resnext.sh"
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} \ 
  /bin/bash -c "cat ${souffle_model_path}/resnext/table4_resnext.csv >> ${script_directory}/../results/table4.csv"

# LSTM
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} \ 
  /bin/bash -c "/workspace/souffle-models/python/models/lstm/run_ncu_lstm.sh"
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} \ 
  /bin/bash -c "cat ${souffle_model_path}/lstm/table4_lstm.csv >> ${script_directory}/../results/table4.csv"

# EfficientNet
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} \ 
  /bin/bash -c "/workspace/souffle-models/python/models/efficientnet/run_ncu_efficientnet.sh"
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} \ 
  /bin/bash -c "cat ${souffle_model_path}/efficientnet/table4_efficientnet.csv >> ${script_directory}/../results/table4.csv"

# SwinTrans.
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} \ 
  /bin/bash -c "/workspace/souffle-models/python/models/swin_transformer/run_ncu_swin_trans.sh"
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} \ 
  /bin/bash -c "cat ${souffle_model_path}/swin_transformer/table4_swin_transformer.csv >> ${script_directory}/../results/table4.csv"

# MMoE
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} \ 
  /bin/bash -c "/workspace/souffle-models/python/models/mmoe/run_ncu_mmoe.sh"
docker exec -it -e SOUFFLE_RUN=${SOUFFLE_RUN} ${souffle_container_id} \ 
  /bin/bash -c "cat ${souffle_model_path}/mmoe/table4_mmoe.csv >> ${script_directory}/../results/table4.csv"

