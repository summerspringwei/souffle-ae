set -xe
cd /workspace/baseline/xla/xla_models
SOUFFLE_RUN=$1
# export TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
# export XLA_FLAGS="--xla_hlo_profile --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_hlo_as_html --xla_dump_hlo_as_proto"
export TF_DUMP_GRAPH_PREFIX="/tmp/tf_dump_graph/"
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
# export XLA_FLAGS="--xla_hlo_profile  --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_hlo_as_html --xla_dump_hlo_as_proto"
# export TF_CPP_MIN_VLOG_LEVEL=2

#  --target-processes all
NCU_ARGS="--metrics dram__bytes_read,gpu__time_duration --clock-control none"

# BERT
NAME=xla_bert
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f \
  python3 tf2load_pb.py --model_file bert-lyj.pb \
  --inputs lyj-input:1,384,768 --outputs layer_0/output/LayerNorm/batchnorm/add_1 --dtype float16 > bert.xla.log 2>&1
sqlite3 --csv xla-bert-nsys.sqlite "${select_latency}" > xla-bert-nsys.xla.csv
fi
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw > ncu-${NAME}.csv
XLA_BERT_MEM=$(python3 ../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
XLA_BERT_NUM_KERNELS=$(wc -l ncu-${NAME}.csv | awk '{ print $1 }')
bert_layer=12
XLA_BERT_MEM=$(python3 -c "print(${XLA_BERT_MEM} * ${bert_layer})")
XLA_BERT_NUM_KERNELS=$(python3 -c "print(${XLA_BERT_NUM_KERNELS} * ${bert_layer})")

# ResNext
NAME=xla_resnext
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f \
  python tf2load_pb.py --model_file resnext_imagenet_101.pb \
  --inputs input:1,3,224,224 --outputs Flatten/flatten/Reshape --dtype float32 > resnext_imagenet_101.xla.log  2>&1
fi
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw | grep -v "*redzone_checker*" > ncu-${NAME}.csv
XLA_RESNEXT_MEM=$(python3 ../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
XLA_RESNEXT_NUM_KERNELS=$(wc -l ncu-${NAME}.csv | awk '{ print $1 }')

# LSTM
NAME=xla_lstm
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f \
  python tf_lstm.py > tf_lstm.xla.log  2>&1
fi
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw > ncu-${NAME}.csv
XLA_LSTM_MEM=$(python3 ../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
XLA_LSTM_NUM_KERNELS=$(wc -l ncu-${NAME}.csv | awk '{ print $1 }')

# EfficientNet
NAME=xla_efficientnet
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f \
  python tf2load_pb.py --model_file efficientnet-b0.pb \
  --inputs input_tensor:1,224,224,3 --outputs efficientnet-b0/model/head/dense/BiasAdd --dtype float32 > efficientnet-b0.xla.log  2>&1
fi
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw > ncu-${NAME}.csv
XLA_EFFICIENTNET_MEM=$(python3 ../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
XLA_EFFICIENTNET_NUM_KERNELS=$(wc -l ncu-${NAME}.csv | awk '{ print $1 }')

# SwinTrans.
NAME=xla_swin_trans
cd Swin-Transformer-Tensorflow
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f \
  python main.py --cfg configs/swin_base_patch4_window7_224.yaml --include_top 1 --resume 1 --weights_type imagenet_1k | grep -v -e "redzone_checker" > swin-trans.xla.log  2>&1
fi
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw > ncu-${NAME}.csv
XLA_SWIN_TRANS_MEM=$(python3 ../../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
XLA_SWIN_TRANS_NUM_KERNELS=$(wc -l ncu-${NAME}.csv | awk '{ print $1 }')
cd ..

# MMOE
NAME=xla_mmoe
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f \
  python3 tf_mmoe.py > tf_mmoe.xla.log  2>&1
fi
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw > ncu-${NAME}.csv
XLA_MMOE_MEM=$(python3 ../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
XLA_MMOE_NUM_KERNELS=$(wc -l ncu-${NAME}.csv | awk '{ print $1 }')


echo "XLA number of kernels:", ${XLA_BERT_NUM_KERNELS}, ${XLA_RESNEXT_NUM_KERNELS}, \
  ${XLA_LSTM_NUM_KERNELS}, ${XLA_EFFICIENTNET_NUM_KERNELS}, \
  ${XLA_SWIN_TRANS_NUM_KERNELS}, ${XLA_MMOE_NUM_KERNELS} > table5_xla.csv
echo "XLA: ", ${XLA_BERT_MEM}, ${XLA_RESNEXT_MEM}, \
  ${XLA_LSTM_MEM}, ${XLA_EFFICIENTNET_MEM}, \
  ${XLA_SWIN_TRANS_MEM}, ${XLA_MMOE_MEM} >> table5_xla_mem.csv
