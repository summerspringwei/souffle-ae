# Run trtexec
export LD_LIBRARY_PATH=/workspace/TensorRT/build:$LD_LIBRARY_PATH
grep_cmd="grep \"\[I\] Latency: min\" | grep -o \"mean = [0-9.]* ms\" | sed 's/mean = //;s/ ms//'"
#BERT
bert_latency=$(/workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/bert_1_384_768_3072_trt_8.4.1_engine.trt 2>/dev/null \
   | grep "\[I\] Latency: min" | grep -o "mean = [0-9.]* ms" | sed 's/mean = //;s/ ms//')
bert_layer=12
TENSORRT_BERT_LATENCY=$(python3 -c "print(${bert_latency} * ${bert_layer})")
# echo ${TENSORRT_BERT_LATENCY}

#ResNext
TENSORRT_RESNEXT_LATENCY=$(
    /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/resnext_imagenet_101_trt_8.4.1_engine.trt 2>/dev/null \
    | grep "\[I\] Latency: min" | grep -o "mean = [0-9.]* ms" | sed 's/mean = //;s/ ms//')
# echo ${resnext_latency}
#LSTM
TENSORRT_LSTM_LATENCY=$(
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/lstm_bs1_h256s100_trt_8.4.1_engine.trt 2>/dev/null \
  | grep "\[I\] Latency: min" | grep -o "mean = [0-9.]* ms" | sed 's/mean = //;s/ ms//')
# echo ${lstm_latency}
#EfficientNet
TENSORRT_EFFICIENTNET_LATENCY=$(
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/efficientnet-b0_trt_8.4.1-engine.trt 2>/dev/null \
  | grep "\[I\] Latency: min" | grep -o "mean = [0-9.]* ms" | sed 's/mean = //;s/ ms//')
# echo ${efficientnet_latency}
#SwinTrans.
TENSORRT_SWIN_TRANS_LATENCY=$(
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/swin-transformer_trt_8.4.1_engine.trt 2>/dev/null \
  | grep "\[I\] Latency: min" | grep -o "mean = [0-9.]* ms" | sed 's/mean = //;s/ ms//')
# echo ${swin_trans_latency}
#MMOE
TENSORRT_MMOE_LATENCY=$(
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/tf_MMoE_1_100_16_8_2_trt_8.4.1_engine.trt 2>/dev/null \
  | grep "\[I\] Latency: min" | grep -o "mean = [0-9.]* ms" | sed 's/mean = //;s/ ms//')
# echo ${mmoe_latency}
echo "TensorRT: ", ${TENSORRT_BERT_LATENCY}, ${TENSORRT_RESNEXT_LATENCY}, \
  ${TENSORRT_LSTM_LATENCY}, ${TENSORRT_EFFICIENTNET_LATENCY}, \
  ${TENSORRT_SWIN_TRANS_LATENCY}, ${TENSORRT_MMOE_LATENCY} | tee /workspace/tensorrt-8.4-engines/table3_tensorrt.csv
