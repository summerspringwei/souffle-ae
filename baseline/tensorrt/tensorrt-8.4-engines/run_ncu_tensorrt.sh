#!/bin/bash 
set -x
# docker cp /home2/xiachunwei/Software/fusion/trt_8.4.1_models ${trt_container_id}:/workspace/
# Run trtexec
#BERT
export LD_LIBRARY_PATH=/workspace/TensorRT/build:$LD_LIBRARY_PATH
TRT_EXEC=/workspace/TensorRT/build/trtexec
MODEL_DIR=/workspace/tensorrt-8.4-engines/

NCU_ARGS="--metrics dram__bytes_read,gpu__time_duration --clock-control none --target-processes all"
TRT_ARGS="--noDataTransfers --warmUp=0 --iterations=1 --duration=0"

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-tensorrt_bert -f \
  ${TRT_EXEC} ${TRT_ARGS} --loadEngine=${MODEL_DIR}/bert_1_384_768_3072_trt_8.4.1_engine.trt 2>/dev/null
fi
ncu -i ./ncu-tensorrt_bert.ncu-rep --csv --page raw | grep -v "void genericReformat::copyPackedKernel"> ncu-tensorrt_bert.csv
TENSORRT_BERT_MEM=$(python3 ${MODEL_DIR}/extract_ncu_cuda_mem_read.py ncu-tensorrt_bert.csv)
TENSORRT_BERT_NUM_KERNELS=$(wc -l ncu-tensorrt_bert.csv | awk '{ print $1 }')
bert_layers=12
TENSORRT_BERT_MEM=$(python3 -c "print(${TENSORRT_BERT_MEM} * ${bert_layers})")
TENSORRT_BERT_NUM_KERNELS=$(python3 -c "print(${TENSORRT_BERT_NUM_KERNELS} * ${bert_layers})")


#ResNext
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-tensorrt_resnext -f \
  ${TRT_EXEC} ${TRT_ARGS}  --loadEngine=${MODEL_DIR}/resnext_imagenet_101_trt_8.4.1_engine.trt 2>/dev/null 
fi
ncu -i ./ncu-tensorrt_resnext.ncu-rep --csv --page raw | grep -v "void genericReformat::copyVectorizedKernel*" | grep -v "void genericReformat::copyPackedKernel" > ncu-tensorrt_resnext.csv
TENSORRT_RESNEXT_MEM=$(python3 ${MODEL_DIR}/extract_ncu_cuda_mem_read.py ncu-tensorrt_resnext.csv)
TENSORRT_RESNEXT_NUM_KERNELS=$(wc -l ncu-tensorrt_resnext.csv | awk '{ print $1 }')

#LSTM
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-tensorrt_lstm -f \
  ${TRT_EXEC} ${TRT_ARGS} --loadEngine=${MODEL_DIR}/lstm_bs1_h256s100_trt_8.4.1_engine.trt 2>/dev/null
fi
ncu -i ./ncu-tensorrt_lstm.ncu-rep --csv --page raw | grep -v "void genericReformat::copyPackedKernel" > ncu-tensorrt_lstm.csv
TENSORRT_LSTM_MEM=$(python3 ${MODEL_DIR}/extract_ncu_cuda_mem_read.py ncu-tensorrt_lstm.csv)
TENSORRT_LSTM_NUM_KERNELS=$(wc -l ncu-tensorrt_lstm.csv | awk '{ print $1 }')


#EfficientNet
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-tensorrt_efficient -f \
  ${TRT_EXEC} ${TRT_ARGS} --loadEngine=${MODEL_DIR}/efficientnet-b0_trt_8.4.1-engine.trt 2>/dev/null
fi
ncu -i ./ncu-tensorrt_efficient.ncu-rep --csv --page raw | grep -v "void genericReformat::copyPackedKernel" > ncu-tensorrt_efficient.csv
TENSORRT_EFFICIENTNET_MEM=$(python3 ${MODEL_DIR}/extract_ncu_cuda_mem_read.py ncu-tensorrt_efficient.csv)
TENSORRT_EFFICIENTNET_NUM_KERNELS=$(wc -l ncu-tensorrt_efficient.csv | awk '{ print $1 }')

#SwinTrans.
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-tensorrt_swin_trans -f \
  ${TRT_EXEC} ${TRT_ARGS} --loadEngine=${MODEL_DIR}/swin-transformer_trt_8.4.1_engine.trt 2>/dev/null
fi
ncu -i ./ncu-tensorrt_swin_trans.ncu-rep --csv --page raw | grep -v "void genericReformat::copyPackedKernel" > ncu-tensorrt_swin_trans.csv
TENSORRT_SWIN_TRANS_MEM=$(python3 ${MODEL_DIR}/extract_ncu_cuda_mem_read.py ncu-tensorrt_swin_trans.csv)
TENSORRT_SWIN_TRANS_NUM_KERNELS=$(wc -l ncu-tensorrt_swin_trans.csv | awk '{ print $1 }')


#MMOE
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-tensorrt_mmoe -f \
  ${TRT_EXEC} ${TRT_ARGS} --loadEngine=${MODEL_DIR}/tf_MMoE_1_100_16_8_2_trt_8.4.1_engine.trt 2>/dev/null
fi
ncu -i ./ncu-tensorrt_mmoe.ncu-rep --csv --page raw | grep -v "void genericReformat::copyPackedKernel" > ncu-tensorrt_mmoe.csv
TENSORRT_MMOE_MEM=$(python3 ${MODEL_DIR}/extract_ncu_cuda_mem_read.py ncu-tensorrt_mmoe.csv)
TENSORRT_MMOE_NUM_KERNELS=$(wc -l ncu-tensorrt_mmoe.csv | awk '{ print $1 }')

echo "TensorRT number of kernels:", ${TENSORRT_BERT_NUM_KERNELS}, ${TENSORRT_RESNEXT_NUM_KERNELS}, \
  ${TENSORRT_LSTM_NUM_KERNELS}, ${TENSORRT_EFFICIENTNET_NUM_KERNELS}, \
  ${TENSORRT_SWIN_TRANS_NUM_KERNELS}, ${TENSORRT_MMOE_NUM_KERNELS} > table5_tensorrt.csv
echo "TensorRT: memory read (MBytes):", ${TENSORRT_BERT_MEM}, ${TENSORRT_RESNEXT_MEM}, \
  ${TENSORRT_LSTM_MEM}, ${TENSORRT_EFFICIENTNET_MEM}, \
  ${TENSORRT_SWIN_TRANS_MEM}, ${TENSORRT_MMOE_MEM} >> table5_tensorrt.csv
