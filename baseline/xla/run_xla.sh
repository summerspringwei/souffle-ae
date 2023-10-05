set -xe
cd /workspace/baseline/xla/xla_models

export TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

select_latency='SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;'

# BERT
nsys profile --stats=true -o xla-bert-nsys -f true \
  python3 tf2load_pb.py --model_file bert-lyj.pb \
  --inputs lyj-input:1,384,768 --outputs layer_0/output/LayerNorm/batchnorm/add_1 --dtype float16 > bert.xla.log 2>&1
sqlite3 --csv xla-bert-nsys.sqlite "${select_latency}" > xla-bert-nsys.xla.csv
python3 extract_nsys_cuda_kernel_latency.py xla-bert-nsys.xla.csv
# ResNext
nsys profile --stats=true -o resnext_imagenet_101-nsys -f true \
  python tf2load_pb.py --model_file resnext_imagenet_101.pb \
  --inputs input:1,3,224,224 --outputs Flatten/flatten/Reshape --dtype float32 > resnext_imagenet_101.xla.log  2>&1
sqlite3 --csv resnext_imagenet_101-nsys.sqlite "${select_latency}" | grep -v -e "redzone_checker" > resnext_imagenet_101-nsys.xla.csv
python3 extract_nsys_cuda_kernel_latency.py resnext_imagenet_101-nsys.xla.csv
# LSTM
nsys profile --stats=true -o tf_lstm-nsys -f true \
  python tf_lstm.py > tf_lstm.xla.log  2>&1
sqlite3 --csv tf_lstm-nsys.sqlite "${select_latency}" | grep -v -e "redzone_checker" > tf_lstm-nsys.xla.csv
python3 extract_nsys_cuda_kernel_latency.py tf_lstm-nsys.xla.csv
# EfficientNet
nsys profile --stats=true -o efficientnet-b0-nsys -f true \
  python tf2load_pb.py --model_file efficientnet-b0.pb \
  --inputs input_tensor:1,224,224,3 --outputs efficientnet-b0/model/head/dense/BiasAdd --dtype float32 | grep -v -e "redzone_checker" > efficientnet-b0.xla.log  2>&1
sqlite3 --csv efficientnet-b0-nsys.sqlite "${select_latency}" > efficientnet-b0-nsys.xla.csv
python3 extract_nsys_cuda_kernel_latency.py efficientnet-b0-nsys.xla.csv
# SwinTrans.
cd Swin-Transformer-Tensorflow
if [ -f models/__init__.py ]; then
    echo "__init__.py exists in the directory."
else
    touch models/__init__.py
fi

ncu --clock-control none -o swin-trans-ncu -f \
    python main.py --cfg configs/swin_base_patch4_window7_224.yaml --include_top 1 --resume 1 --weights_type imagenet_1k | grep -v -e "redzone_checker" | tee swin-trans.xla.log  2>&1
ncu -i swin-trans-ncu.ncu-rep --csv --page raw > swin-trans-ncu.xla.csv
python3 
# nsys profile --stats=true -o swin-trans-nsys -f true \
#   python main.py --cfg configs/swin_base_patch4_window7_224.yaml --include_top 1 --resume 1 --weights_type imagenet_1k | grep -v -e "redzone_checker" > swin-trans.xla.log  2>&1
# sqlite3 --csv swin-trans-nsys.sqlite "${select_latency}" > swin-trans-nsys.xla.csv
# python3 ../extract_nsys_cuda_kernel_latency.py swin-trans-nsys.xla.csv
python3 ../../../extract_ncu_cuda_kernel_latency.py swin-trans-ncu.xla.csv
cd ..
# MMOE
nsys profile --stats=true -o tf_mmoe-nsys -f true \
  python3 tf_mmoe.py > tf_mmoe.xla.log  2>&1
sqlite3 --csv tf_mmoe-nsys.sqlite "${select_latency}" | grep -v -e "redzone_checker" > tf_mmoe-nsys.xla.csv
python3 extract_nsys_cuda_kernel_latency.py tf_mmoe-nsys.xla.csv
