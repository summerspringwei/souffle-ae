set -xe
cd /workspace/xla_models

select_latency='SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;'

# BERT
nsys profile --stats=true -o xla-bert-nsys -f true \
  python3 tf2load_pb.py --model_file bert-lyj.pb \
  --inputs lyj-input:1,384,768 --outputs layer_0/output/LayerNorm/batchnorm/add_1 --dtype float16 > xla-bert.log 2>&1
sqlite3 --csv xla-bert-nsys.sqlite "${select_latency}" > xla-bert-nsys.csv
python3 extract_nsys_cuda_kernel_latency.py xla-bert-nsys.csv
# ResNext
nsys profile --stats=true -o resnext_imagenet_101-nsys -f true \
  python tf2load_pb.py --model_file resnext_imagenet_101.pb \
  --inputs input:1,3,224,224 --outputs Flatten/flatten/Reshape --dtype float32 > resnext_imagenet_101.log  2>&1
sqlite3 --csv resnext_imagenet_101-nsys.sqlite "${select_latency}" | grep -v -e "redzone_checker" > resnext_imagenet_101-nsys.csv
python3 extract_nsys_cuda_kernel_latency.py resnext_imagenet_101-nsys.csv
# LSTM
nsys profile --stats=true -o tf_lstm-nsys -f true \
  python tf_lstm.py > tf_lstm.log  2>&1
sqlite3 --csv tf_lstm-nsys.sqlite "${select_latency}" | grep -v -e "redzone_checker" > tf_lstm-nsys.csv
python3 extract_nsys_cuda_kernel_latency.py tf_lstm-nsys.csv
# EfficientNet
nsys profile --stats=true -o efficientnet-b0-nsys -f true \
  python tf2load_pb.py --model_file efficientnet-b0.pb \
  --inputs input_tensor:1,224,224,3 --outputs efficientnet-b0/model/head/dense/BiasAdd --dtype float32 | grep -v -e "redzone_checker" > efficientnet-b0.log  2>&1
sqlite3 --csv efficientnet-b0-nsys.sqlite "${select_latency}" > efficientnet-b0-nsys.csv
python3 extract_nsys_cuda_kernel_latency.py efficientnet-b0-nsys.csv
# SwinTrans.
cd Swin-Transformer-Tensorflow
nsys profile --stats=true -o swin-trans-nsys -f true \
  python main.py --cfg configs/swin_base_patch4_window7_224.yaml --include_top 1 --resume 1 --weights_type imagenet_1k | grep -v -e "redzone_checker" > swin-trans.log  2>&1
sqlite3 --csv swin-trans-nsys.sqlite "${select_latency}" > swin-trans-nsys.csv
python3 ../extract_nsys_cuda_kernel_latency.py swin-trans-nsys.csv
cd ..
# MMOE
nsys profile --stats=true -o tf_mmoe-nsys -f true \
  python3 tf_mmoe.py > tf_mmoe.log  2>&1
sqlite3 --csv tf_mmoe-nsys.sqlite "${select_latency}" | grep -v -e "redzone_checker" > tf_mmoe-nsys.csv
python3 extract_nsys_cuda_kernel_latency.py tf_mmoe-nsys.csv
