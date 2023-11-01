
set -x
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

# ResNext
NAME=xla_resnext
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f \
  python tf2load_pb.py --model_file resnext_imagenet_101.pb \
  --inputs input:1,3,224,224 --outputs Flatten/flatten/Reshape --dtype float32 > resnext_imagenet_101.xla.log  2>&1
fi
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw \
  | grep -v "redzone_checker" \
  | grep -v "void convolve_common_engine_float_NHWC"\
  | grep -v "void cudnn::ops::nchwToNhwcKernel" \
  | grep -v "__xla_fp32_comparison"\
  | grep -v "void cudnn::ops::nhwcToNchwKernel" > ncu-${NAME}.csv
XLA_RESNEXT_MEM=$(python3 ../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
XLA_RESNEXT_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-${NAME}.csv)
XLA_RESNEXT_NUM_KERNELS=$(wc -l ncu-${NAME}.csv | awk '{ print $1 }')

# EfficientNet
NAME=xla_efficientnet
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f \
  python tf2load_pb.py --model_file efficientnet-b0.pb \
  --inputs input_tensor:1,224,224,3 --outputs efficientnet-b0/model/head/dense/BiasAdd --dtype float32 > efficientnet-b0.xla.log  2>&1
fi
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw \
  | grep -v "redzone_checker" \
  | grep -v "void convolve_common_engine_float_NHWC"\
  | grep -v "__xla_fp32_comparison"\
  | grep -v "void cudnn::ops::nhwcToNchwKernel" \
  | grep -v "void cudnn::ops::nchwToNhwcKernel" > ncu-${NAME}.csv
XLA_EFFICIENTNET_MEM=$(python3 ../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
XLA_EFFICIENTNET_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-${NAME}.csv)
XLA_EFFICIENTNET_NUM_KERNELS=$(wc -l ncu-${NAME}.csv | awk '{ print $1 }')

echo "XLA RESNEXT: latency: ${XLA_RESNEXT_LATENCY}, memory read: ${XLA_RESNEXT_MEM}, number of kernels ${XLA_RESNEXT_NUM_KERNELS}"
echo "XLA EFFICIENTNET: latency: ${XLA_EFFICIENTNET_LATENCY}, memory read: ${XLA_EFFICIENTNET_MEM}, number of kernels ${XLA_EFFICIENTNET_NUM_KERNELS}"
