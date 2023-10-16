set -xe
select_latency='SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;'
script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${script_directory}
# BERT
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o iree-bert-nsys -f true \
  /workspace/iree-build/tools/iree-run-module --device=cuda --module=bert/bert_1_384_768_3072.vmfb  --input="1x384x768xf16=0" > iree-bert.log 2>&1
sqlite3 --csv iree-bert-nsys.sqlite "${select_latency}" > iree-bert-nsys.csv
fi
IREE_BERT_LATENCY=$(python3 ${script_directory}/extract_nsys_cuda_kernel_latency.py iree-bert-nsys.csv)
bert_layer=12
IREE_BERT_LATENCY=$(python3 -c "print(${IREE_BERT_LATENCY} * ${bert_layer})")

# ResNext
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o iree-resnext_imagenet_101-nsys -f true \
  /workspace/iree-build/tools/iree-run-module --device=cuda --module=resnext/resnext_imagenet_101.vmfb  --input="1x3x224x224xf32=0"
sqlite3 --csv iree-resnext_imagenet_101-nsys.sqlite "${select_latency}" > iree-resnext_imagenet_101-nsys.csv
fi
IREE_RESNEXT_LATENCY=$(python3 ${script_directory}/extract_nsys_cuda_kernel_latency.py iree-resnext_imagenet_101-nsys.csv)

# LSTM
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o iree-tf_lstm-nsys -f true \
  /workspace/iree-build/tools/iree-run-module --device=cuda --module=lstm/frozen_lstm_l8s8h256_bs1.vmfb  --input="8x1x256xf32=0"
sqlite3 --csv iree-tf_lstm-nsys.sqlite "${select_latency}" > iree-tf_lstm-nsys.csv
fi
IREE_LSTM_LATENCY=$(python3 ${script_directory}/extract_nsys_cuda_kernel_latency.py iree-tf_lstm-nsys.csv)

# EfficientNet
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o iree-efficientnet-b0-nsys -f true \
  /workspace/iree-build/tools/iree-run-module --device=cuda --module=efficientnet/efficientnet-b0.vmfb  --input="1x3x224x224xf32=0"
sqlite3 --csv iree-efficientnet-b0-nsys.sqlite "${select_latency}" > iree-efficientnet-b0-nsys.csv
fi
IREE_EFFICIENTNET_LATENCY=$(python3 ${script_directory}/extract_nsys_cuda_kernel_latency.py iree-efficientnet-b0-nsys.csv)

# SwinTrans.
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o iree-swin-transformer-nsys -f true \
  /workspace/iree-build/tools/iree-run-module --device=cuda --module=swin-transformer/swin-transformer.vmfb  --input="1x3x224x224xf32=0"
sqlite3 --csv iree-swin-transformer-nsys.sqlite "${select_latency}" > iree-swin-transformer-nsys.csv
fi
IREE_SWIN_TRANS_LATENCY=$(python3 ${script_directory}/extract_nsys_cuda_kernel_latency.py iree-swin-transformer-nsys.csv)

# MMOE
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o iree-tf_MMoE_1_100_16_8_2-nsys -f true \
  /workspace/iree-build/tools/iree-run-module --device=cuda --module=mmoe/tf_MMoE_1_100_16_8_2.vmfb  --input="1x100xf32=0"
sqlite3 --csv iree-tf_MMoE_1_100_16_8_2-nsys.sqlite "${select_latency}" > iree-tf_MMoE_1_100_16_8_2-nsys.csv
fi
IREE_MMoE_LATENCY=$(python3 ${script_directory}/extract_nsys_cuda_kernel_latency.py iree-tf_MMoE_1_100_16_8_2-nsys.csv)

echo "IREE: ", ${IREE_BERT_LATENCY}, ${IREE_RESNEXT_LATENCY}, \
  ${IREE_LSTM_LATENCY}, ${IREE_EFFICIENTNET_LATENCY}, \
  ${IREE_SWIN_TRANS_LATENCY}, ${IREE_MMoE_LATENCY} | tee table3_iree.csv
