#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:/usr/local/cuda-11.1/targets/x86_64-linux/lib/stubs:/usr/local/cuda-11.1/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
set -xe
select_latency='SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;'
MINDSPORE_FOLDER=/workspace/baseline/mindspore
cd ${MINDSPORE_FOLDER=/workspace/baseline/mindspore}
# BERT
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o apollo-bert-nsys -f true \
  python3 /workspace/mindspore/model_zoo/official/nlp/bert/src/my_run_bert.py > bert.apollo.log
  sqlite3 --csv apollo-bert-nsys.sqlite "${select_latency}" > apollo-bert-nsys.csv
fi
APOLLO_BERT_LATENCY=$(python3 ${MINDSPORE_FOLDER}/extract_nsys_cuda_kernel_latency.py apollo-bert-nsys.csv)

# ResNext
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o apollo-resnext_imagenet_101-nsys -f true \
  python3 /workspace/mindspore_models/official/cv/resnext/my_run_resnext101.py > resnext.apollo.log
sqlite3 --csv apollo-resnext_imagenet_101-nsys.sqlite "${select_latency}" > apollo-resnext_imagenet_101-nsys.csv
fi
APOLLO_RESNEXT_LATENCY=$(python3 ${MINDSPORE_FOLDER}/extract_nsys_cuda_kernel_latency.py apollo-resnext_imagenet_101-nsys.csv)

# LSTM
# Failed
APOLLO_LSTM_LATENCY="Failed"

# EfficientNet
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o apollo-efficientnet-b0-nsys -f true \
  python3 /workspace/mindspore_models/official/cv/efficientnet/my_run_efficientnet.py \
  --config_path /workspace/mindspore_models/official/cv/efficientnet/efficientnet_b0_imagenet_config.yaml > efficient.apollo.log
  sqlite3 --csv apollo-efficientnet-b0-nsys.sqlite "${select_latency}" > apollo-efficientnet-b0-nsys.csv
fi
APOLLO_EFFICIENTNET_LATENCY=$(python3 ${MINDSPORE_FOLDER}/extract_nsys_cuda_kernel_latency.py apollo-efficientnet-b0-nsys.csv)

# SwinTrans.
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o apollo-swin-transformer-nsys -f true \
  python3 /workspace/mindspore_models/research/cv/swin_transformer/my_run_swin_transformer.py \
  --swin_config /workspace/mindspore_models/research/cv/swin_transformer/src/configs/swin_base_patch4_window7_224.yaml > swin_transformer.apollo.log
  sqlite3 --csv apollo-swin-transformer-nsys.sqlite "${select_latency}" > apollo-swin-transformer-nsys.csv
fi

APOLLO_SWIN_TRANS_LATENCY=$(python3 ${MINDSPORE_FOLDER}/extract_nsys_cuda_kernel_latency.py apollo-swin-transformer-nsys.csv)

# MMOE
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o apollo-tf_MMoE_1_100_16_8_2-nsys -f true \
  python3 /workspace/mindspore/model_zoo/official/nlp/bert/src/mindspore_mmoe.py > mmoe.apollo.log
  sqlite3 --csv apollo-tf_MMoE_1_100_16_8_2-nsys.sqlite "${select_latency}" > apollo-tf_MMoE_1_100_16_8_2-nsys.csv
fi
APOLLO_MMoE_LATENCY=$(python3 ${MINDSPORE_FOLDER}/extract_nsys_cuda_kernel_latency.py apollo-tf_MMoE_1_100_16_8_2-nsys.csv)

# echo "Apollo: " ${APOLLO_BERT_LATENCY}, ${APOLLO_RESNEXT_LATENCY}, ${APOLLO_LSTM_LATENCY}, ${APOLLO_EFFICIENTNET_LATENCY}, ${APOLLO_SWIN_TRANS_LATENCY}, ${APOLLO_MMoE_LATENCY} | tee table3_apollo.csv

python3 -c "print('Apollo:, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(${APOLLO_BERT_LATENCY}, ${APOLLO_RESNEXT_LATENCY}, ${APOLLO_LSTM_LATENCY}, ${APOLLO_EFFICIENTNET_LATENCY}, ${APOLLO_SWIN_TRANS_LATENCY}, ${APOLLO_MMoE_LATENCY}))" | tee table3_apollo.csv
