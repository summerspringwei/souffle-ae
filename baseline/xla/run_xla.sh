set -xe
cd /workspace/baseline/xla/xla_models

# export TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
# export XLA_FLAGS="--xla_hlo_profile --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_hlo_as_html --xla_dump_hlo_as_proto"
export TF_DUMP_GRAPH_PREFIX="/tmp/tf_dump_graph/"
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
# export XLA_FLAGS="--xla_hlo_profile  --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_hlo_as_html --xla_dump_hlo_as_proto"
# export TF_CPP_MIN_VLOG_LEVEL=2

select_latency='SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;'

# BERT
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o xla-bert-nsys -f true \
  python3 tf2load_pb.py --model_file bert-lyj.pb \
  --inputs lyj-input:1,384,768 --outputs layer_0/output/LayerNorm/batchnorm/add_1 --dtype float16 > bert.xla.log 2>&1
sqlite3 --csv xla-bert-nsys.sqlite "${select_latency}" > xla-bert-nsys.xla.csv
fi
XLA_BERT_LATENCY=$(python3 extract_nsys_cuda_kernel_latency.py xla-bert-nsys.xla.csv)
bert_layer=12
XLA_BERT_LATENCY=$(python3 -c "print(${XLA_BERT_LATENCY} * ${bert_layer})")

# ResNext
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o resnext_imagenet_101-nsys -f true \
  python tf2load_pb.py --model_file resnext_imagenet_101.pb \
  --inputs input:1,3,224,224 --outputs Flatten/flatten/Reshape --dtype float32 > resnext_imagenet_101.xla.log  2>&1
sqlite3 --csv resnext_imagenet_101-nsys.sqlite "${select_latency}" | grep -v -e "redzone_checker" > resnext_imagenet_101-nsys.xla.csv
fi
XLA_RESNEXT_LATENCY=$(python3 extract_nsys_cuda_kernel_latency.py resnext_imagenet_101-nsys.xla.csv)

# LSTM
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o tf_lstm-nsys -f true \
  python tf_lstm.py > tf_lstm.xla.log  2>&1
sqlite3 --csv tf_lstm-nsys.sqlite "${select_latency}" | grep -v -e "redzone_checker" > tf_lstm-nsys.xla.csv
fi
XLA_LSTM_LATENCY=$(python3 extract_nsys_cuda_kernel_latency.py tf_lstm-nsys.xla.csv)

# EfficientNet
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o efficientnet-b0-nsys -f true \
  python tf2load_pb.py --model_file efficientnet-b0.pb \
  --inputs input_tensor:1,224,224,3 --outputs efficientnet-b0/model/head/dense/BiasAdd --dtype float32 > efficientnet-b0.xla.log  2>&1
sqlite3 --csv efficientnet-b0-nsys.sqlite "${select_latency}" | grep -v -e "redzone_checker"  > efficientnet-b0-nsys.xla.csv
fi
XLA_EFFICIENT_LATENCY=$(python3 extract_nsys_cuda_kernel_latency.py efficientnet-b0-nsys.xla.csv)

# SwinTrans.
cd Swin-Transformer-Tensorflow
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
# ncu --clock-control none -o swin-trans-ncu -f \
#     python main.py --cfg configs/swin_base_patch4_window7_224.yaml --include_top 1 --resume 1 --weights_type imagenet_1k > swin-trans.xla.log  2>&1
# ncu -i swin-trans-ncu.ncu-rep --csv --page raw | grep -v -e "redzone_checker" > swin-trans-ncu-raw.xla.csv
nsys profile --stats=true -o swin-trans-nsys -f true \
  python main.py --cfg configs/swin_base_patch4_window7_224.yaml --include_top 1 --resume 1 --weights_type imagenet_1k | grep -v -e "redzone_checker" > swin-trans.xla.log  2>&1
sqlite3 --csv swin-trans-nsys.sqlite "${select_latency}" > swin-trans-nsys.xla.csv
fi
python3 ../extract_nsys_cuda_kernel_latency.py swin-trans-nsys.xla.csv
XLA_SWIN_TRANS_LATENCY=$(python3 ../../../extract_ncu_cuda_kernel_latency.py swin-trans-ncu-raw.xla.csv)
cd ..

# MMOE
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
nsys profile --stats=true -o tf_mmoe-nsys -f true \
  python3 tf_mmoe.py > tf_mmoe.xla.log  2>&1
sqlite3 --csv tf_mmoe-nsys.sqlite "${select_latency}" | grep -v -e "redzone_checker" > tf_mmoe-nsys.xla.csv
fi
XLA_MMoE_LATENCY=$(python3 extract_nsys_cuda_kernel_latency.py tf_mmoe-nsys.xla.csv)

# echo "XLA: ," ${XLA_BERT_LATENCY}, ${XLA_RESNEXT_LATENCY}, \
#   ${XLA_LSTM_LATENCY}, ${XLA_EFFICIENT_LATENCY}, \
#   ${XLA_SWIN_TRANS_LATENCY}, ${XLA_MMoE_LATENCY} | tee table3_xla.csv

python3 -c "print('XLA:, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(${XLA_BERT_LATENCY}, ${XLA_RESNEXT_LATENCY}, ${XLA_LSTM_LATENCY}, ${XLA_EFFICIENT_LATENCY}, ${XLA_SWIN_TRANS_LATENCY}, ${XLA_MMoE_LATENCY}))"  | tee table3_xla.csv
