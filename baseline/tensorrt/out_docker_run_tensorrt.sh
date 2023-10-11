trt_container_id=$(docker ps -qf "ancestor=souffle-tensorrt8.4.1-ubuntu18.04:latest")
# docker cp /home2/xiachunwei/Software/fusion/trt_8.4.1_models ${trt_container_id}:/workspace/
# Run trtexec
#BERT
bert_latency=$(docker exec -it ${trt_container_id} \
  /bin/bash -c \
  "export LD_LIBRARY_PATH=/workspace/TensorRT/build:$LD_LIBRARY_PATH && \
   /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/trt_8.4.1_models/bert_1_384_768_3072_trt_8.4.1_engine.trt 2>/dev/null\
   | grep \"\[I\] Latency: min\" | grep -o \"mean = [0-9.]* ms\" | sed 's/mean = //;s/ ms//'")
bert_layer=12
TENSORRT_BERT_LATENCY=$(python3 -c "print(${bert_latency} * ${bert_layer})")

#ResNext
TENSORRT_RESNEXT_LATENCY=$(docker exec -it ${trt_container_id} \
  /bin/bash -c \
  "export LD_LIBRARY_PATH=/workspace/TensorRT/build:$LD_LIBRARY_PATH && \
    /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/trt_8.4.1_models/resnext_imagenet_101_trt_8.4.1_engine.trt 2>/dev/null\
    | grep \"\[I\] Latency: min\" | grep -o \"mean = [0-9.]* ms\" | sed 's/mean = //;s/ ms//'")

#LSTM
TENSORRT_LSTM_LATENCY=$(docker exec -it ${trt_container_id} \
  /bin/bash -c \
  "export LD_LIBRARY_PATH=/workspace/TensorRT/build:$LD_LIBRARY_PATH && \
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/trt_8.4.1_models/lstm_bs1_h256s100_trt_8.4.1_engine.trt 2>/dev/null\
  | grep \"\[I\] Latency: min\" | grep -o \"mean = [0-9.]* ms\" | sed 's/mean = //;s/ ms//'")

#EfficientNet
TENSORRT_EFFICIENTNET_LATENCY=$(docker exec -it ${trt_container_id} \
  /bin/bash -c \
  "export LD_LIBRARY_PATH=/workspace/TensorRT/build:$LD_LIBRARY_PATH && \
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/trt_8.4.1_models/efficientnet-b0_trt_8.4.1-engine.trt 2>/dev/null\
  | grep \"\[I\] Latency: min\" | grep -o \"mean = [0-9.]* ms\" | sed 's/mean = //;s/ ms//'")

#SwinTrans.
TENSORRT_SWIN_TRANS_LATENCY=$(docker exec -it ${trt_container_id} \
  /bin/bash -c \
  "export LD_LIBRARY_PATH=/workspace/TensorRT/build:$LD_LIBRARY_PATH && \
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/trt_8.4.1_models/swin-transformer_trt_8.4.1_engine.trt 2>/dev/null\
  | grep \"\[I\] Latency: min\" | grep -o \"mean = [0-9.]* ms\" | sed 's/mean = //;s/ ms//'")

#MMOE
TENSORRT_MMOE_LATENCY=$(docker exec -it ${trt_container_id} \
  /bin/bash -c \
  "export LD_LIBRARY_PATH=/workspace/TensorRT/build:$LD_LIBRARY_PATH && \
  /workspace/TensorRT/build/trtexec --noDataTransfers --loadEngine=/workspace/trt_8.4.1_models/tf_MMoE_1_100_16_8_2_trt_8.4.1_engine.trt 2>/dev/null\
  | grep \"\[I\] Latency: min\" | grep -o \"mean = [0-9.]* ms\" | sed 's/mean = //;s/ ms//'")

# echo ${bert_latency} ${resnext_latency} ${lstm_latency} ${efficientnet_latency} ${swin_trans_latency} ${mmoe_latency}
echo "TensorRT: ", ${TENSORRT_BERT_LATENCY}, ${resnext_latency}, \
  ${TENSORRT_LSTM_LATENCY}, ${TENSORRT_EFFICIENTNET_LATENCY}, \
  ${TENSORRT_SWIN_TRANS_LATENCY}, ${TENSORRT_MMOE_LATENCY} | tee table3_tensorrt.csv