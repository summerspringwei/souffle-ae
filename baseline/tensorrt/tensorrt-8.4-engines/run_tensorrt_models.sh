# Run trtexec
export LD_LIBRARY_PATH=/workspace/TensorRT/build:$LD_LIBRARY_PATH
grep_cmd="grep \"\[I\] Latency: min\" | grep -o \"mean = [0-9.]* ms\" | sed 's/mean = //;s/ ms//'"
#BERT
bert_latency=$(/workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/bert_1_384_768_3072_trt_8.4.1_engine.trt 2>/dev/null \
   | grep "\[I\] Latency: min" | grep -o "mean = [0-9.]* ms" | sed 's/mean = //;s/ ms//')
bert_layer=12
bert_latency=$(python3 -c "print(${bert_latency} * ${bert_layer})")
echo ${bert_latency}

#ResNext
resnext_latency=$(
    /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/resnext_imagenet_101_trt_8.4.1_engine.trt 2>/dev/null \
    | grep "\[I\] Latency: min" | grep -o "mean = [0-9.]* ms" | sed 's/mean = //;s/ ms//')
echo ${resnext_latency}
#LSTM
lstm_latency=$(
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/lstm_bs1_h256s100_trt_8.4.1_engine.trt 2>/dev/null \
  | grep "\[I\] Latency: min" | grep -o "mean = [0-9.]* ms" | sed 's/mean = //;s/ ms//')
echo ${lstm_latency}
#EfficientNet
efficientnet_latency=$(
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/efficientnet-b0_trt_8.4.1-engine.trt 2>/dev/null \
  | grep "\[I\] Latency: min" | grep -o "mean = [0-9.]* ms" | sed 's/mean = //;s/ ms//')
echo ${efficientnet_latency}
#SwinTrans.
swin_trans_latency=$(
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/swin-transformer_trt_8.4.1_engine.trt 2>/dev/null \
  | grep "\[I\] Latency: min" | grep -o "mean = [0-9.]* ms" | sed 's/mean = //;s/ ms//')
echo ${swin_trans_latency}
#MMOE
mmoe_latency=$(
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/tensorrt-8.4-engines/tf_MMoE_1_100_16_8_2_trt_8.4.1_engine.trt 2>/dev/null \
  | grep "\[I\] Latency: min" | grep -o "mean = [0-9.]* ms" | sed 's/mean = //;s/ ms//')
echo ${mmoe_latency}
echo ${bert_latency} ${resnext_latency} ${lstm_latency} ${efficientnet_latency} ${swin_trans_latency} ${mmoe_latency}
