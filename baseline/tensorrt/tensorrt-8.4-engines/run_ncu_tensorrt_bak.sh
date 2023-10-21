# Run trtexec
export LD_LIBRARY_PATH=/workspace/TensorRT/build:$LD_LIBRARY_PATH
#BERT
# ncu --clock-control none -o tensorrt-bert-ncu -f --set detailed \
#   /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/bert_1_384_768_3072_trt_8.4.1_engine.trt
# ncu -i tensorrt-bert-ncu.ncu-rep --page raw --csv > tensorrt-bert-ncu.csv
# python3 extract_ncu_cuda_kernel_latency.py tensorrt-bert-ncu.csv
# bert_layer=12
# TENSORRT_BERT_LATENCY=$(python3 -c "print(${bert_latency} * ${bert_layer})")

#ResNext
ncu --clock-control none -o tensorrt-resnext-ncu -f --set detailed \
    /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/resnext_imagenet_101_trt_8.4.1_engine.trt
ncu -i tensorrt-resnext-ncu.ncu-rep --page raw --csv > tensorrt-resnext-ncu.csv

#LSTM
ncu --clock-control none -o tensorrt-lstm-ncu -f --set detailed \
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/lstm_bs1_h256s100_trt_8.4.1_engine.trt
ncu -i tensorrt-lstm-ncu.ncu-rep --page raw --csv > tensorrt-lstm-ncu.csv

#EfficientNet
ncu --clock-control none -o tensorrt-efficientnet-ncu -f --set detailed \
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/efficientnet-b0_trt_8.4.1-engine.trt
ncu -i tensorrt-efficientnet-ncu.ncu-rep --page raw --csv > tensorrt-efficientnet-ncu.csv

#SwinTrans.
ncu --clock-control none -o tensorrt-swin_tran-ncu -f --set detailed \
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/swin-transformer_trt_8.4.1_engine.trt
ncu -i tensorrt-swin_tran-ncu.ncu-rep --page raw --csv > tensorrt-swin_tran-ncu.csv

#MMOE
ncu --clock-control none -o tensorrt-mmoe-ncu -f --set detailed \
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/tf_MMoE_1_100_16_8_2_trt_8.4.1_engine.trt
ncu -i tensorrt-mmoe-ncu.ncu-rep --page raw --csv > tensorrt-mmoe-ncu.csv

echo "TensorRT: ", ${TENSORRT_BERT_LATENCY}, ${TENSORRT_RESNEXT_LATENCY}, \
  ${TENSORRT_LSTM_LATENCY}, ${TENSORRT_EFFICIENTNET_LATENCY}, \
  ${TENSORRT_SWIN_TRANS_LATENCY}, ${TENSORRT_MMOE_LATENCY} | tee table3_tensorrt.csv
