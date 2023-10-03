#!/bin/bash
set -xe
select_latency='SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;'

# BERT
nsys profile --stats=true -o apollo-bert-nsys -f true \
  python3 /workspace/mindspore/model_zoo/official/nlp/bert/src/my_run_bert.py > bert_tmp.txt
sqlite3 --csv apollo-bert-nsys.sqlite "${select_latency}" > apollo-bert-nsys.csv
python3 extract_nsys_cuda_kernel_latency.py apollo-bert-nsys.csv

# ResNext
nsys profile --stats=true -o iree-resnext_imagenet_101-nsys -f true \
  python3 /workspace/model_zoo_mindspore/models/official/cv/resnext/my_run_resnext101.py > resnext_tmp.txt
sqlite3 --csv iree-resnext_imagenet_101-nsys.sqlite "${select_latency}" > iree-resnext_imagenet_101-nsys.csv
python3 extract_nsys_cuda_kernel_latency.py iree-resnext_imagenet_101-nsys.csv

# LSTM
# Failed

# EfficientNet
nsys profile --stats=true -o iree-efficientnet-b0-nsys -f true \
  python3 /workspace/model_zoo_mindspore/models/official/cv/efficientnet/my_run_efficientnet.py --config_path efficientnet_b0_imagenet_config.yaml > efficient_net_tmp.txt
sqlite3 --csv iree-efficientnet-b0-nsys.sqlite "${select_latency}" > iree-efficientnet-b0-nsys.csv
python3 extract_nsys_cuda_kernel_latency.py iree-efficientnet-b0-nsys.csv

# SwinTrans.
nsys profile --stats=true -o iree-swin-transformer-nsys -f true \
  python3 /workspace/model_zoo_mindspore/models/research/cv/swin_transformer/my_run_swin_transformer.py --swin_config src/configs/swin_base_patch4_window7_224.yaml > swin_transformer_tmp.txt
sqlite3 --csv iree-swin-transformer-nsys.sqlite "${select_latency}" > iree-swin-transformer-nsys.csv
python3 extract_nsys_cuda_kernel_latency.py iree-swin-transformer-nsys.csv

# MMOE
nsys profile --stats=true -o iree-tf_MMoE_1_100_16_8_2-nsys -f true \
  python3 /workspace/mindspore/model_zoo/official/nlp/bert/src/mindspore_mmoe.py > mmoe_tmp.txt
sqlite3 --csv iree-tf_MMoE_1_100_16_8_2-nsys.sqlite "${select_latency}" > iree-tf_MMoE_1_100_16_8_2-nsys.csv
python3 extract_nsys_cuda_kernel_latency.py iree-tf_MMoE_1_100_16_8_2-nsys.csv
