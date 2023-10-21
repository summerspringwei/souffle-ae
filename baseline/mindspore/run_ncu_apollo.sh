#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:/usr/local/cuda-11.1/targets/x86_64-linux/lib/stubs:/usr/local/cuda-11.1/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
set -xe

NCU_ARGS="--metrics dram__bytes_read,gpu__time_duration --clock-control none --target-processes all"

MINDSPORE_FOLDER=/workspace/baseline/mindspore
cd ${MINDSPORE_FOLDER=/workspace/baseline/mindspore}
# BERT
# NAME=apollo_bert
# if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
# ncu ${NCU_ARGS} -o ncu-${NAME} -f \
#   python3 /workspace/mindspore/model_zoo/official/nlp/bert/src/my_run_bert.py > bert.apollo.log
# fi
# ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw | grep -v "void CastKernel*" > ncu-${NAME}.csv
# APOLLO_BERT_MEM=$(python3 extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
# APOLLO_BERT_NUM_KERNELS=$(wc -l ncu-${NAME}.csv | awk '{ print $1 }')

# ResNext
# NAME=apollo_resnext
# if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
# ncu ${NCU_ARGS} -o ncu-${NAME} -f \
#   python3 /workspace/mindspore_models/official/cv/resnext/my_run_resnext101.py > resnext.apollo.log
# fi
# ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw > ncu-${NAME}.csv
# APOLLO_RESNEXT_MEM=$(python3 extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
# APOLLO_RESNEXT_NUM_KERNELS=$(wc -l ncu-${NAME}.csv | awk '{ print $1 }')

# LSTM
# Failed
APOLLO_LSTM_MEM="Failed"
APOLLO_LSTM_NUM_KERNELS="Failed"

# EfficientNet
# NAME=apollo_efficientnet
# if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
# ncu ${NCU_ARGS} -o ncu-${NAME} -f \
#   python3 /workspace/mindspore_models/official/cv/efficientnet/my_run_efficientnet.py \
#   --config_path /workspace/mindspore_models/official/cv/efficientnet/efficientnet_b0_imagenet_config.yaml > efficient.apollo.log
# fi
# ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw > ncu-${NAME}.csv
# APOLLO_EFFICIENTNET_MEM=$(python3 extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
# APOLLO_EFFICIENTNET_NUM_KERNELS=$(wc -l ncu-${NAME}.csv | awk '{ print $1 }')


# SwinTrans.
NAME=apollo_swin_trans
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f \
  python3 /workspace/mindspore_models/research/cv/swin_transformer/my_run_swin_transformer.py \
  --swin_config /workspace/mindspore_models/research/cv/swin_transformer/src/configs/swin_base_patch4_window7_224.yaml > swin_transformer.apollo.log
fi
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw > ncu-${NAME}.csv
APOLLO_SWIN_TRANS_MEM=$(python3 extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
APOLLO_SWIN_TRANS_NUM_KERNELS=$(wc -l ncu-${NAME}.csv | awk '{ print $1 }')
exit 0
# MMOE
NAME=apollo_mmoe
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f \
  python3 /workspace/mindspore/model_zoo/official/nlp/bert/src/mindspore_mmoe.py > mmoe.apollo.log
fi
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw > ncu-${NAME}.csv
APOLLO_MMOE_MEM=$(python3 extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
APOLLO_MMOE_NUM_KERNELS=$(wc -l ncu-${NAME}.csv | awk '{ print $1 }')

echo "Apollo: ", ${APOLLO_BERT_NUM_KERNELS}, ${APOLLO_RESNEXT_NUM_KERNELS}, \
  ${APOLLO_LSTM_NUM_KERNELS}, ${APOLLO_EFFICIENTNET_NUM_KERNELS}, \
  ${APOLLO_SWIN_TRANS_NUM_KERNELS}, ${APOLLO_MMOE_NUM_KERNELS} > table5_apollo.csv
echo "Apollo: ", ${APOLLO_BERT_MEM}, ${APOLLO_RESNEXT_MEM}, \
  ${APOLLO_LSTM_MEM}, ${APOLLO_EFFICIENTNET_MEM}, \
  ${APOLLO_SWIN_TRANS_MEM}, ${APOLLO_MMOE_MEM} >> table5_apollo.csv
