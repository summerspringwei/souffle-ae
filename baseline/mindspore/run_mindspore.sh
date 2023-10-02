# 跑模型的时候新开一个窗口，用 nvidia-smi dmon 监测一下GPU频率，看看是不是跑到了1410MHz

#先锁定GPU 频率
sudo nvidia-smi -lgc 1410,1410

#BERT 模型
# 运行BERT base，先到BERT的文件夹
cd ~/Software/mindspore/model_zoo/official/nlp/bert/src
export GLOG_v=0 && python3 my_run_bert.py > bert_tmp.txt 2>&1
# Grep 测量的每个op的时间
cat bert_tmp.txt | grep "xiachunwei-latency"
# Grep 出来op的时间
cat bert_tmp.txt | grep "xiachunwei-latency"| awk '{ print $8 }'
# 记录一下总的时间和op的数量到表格里

# ResNext-101
cd /home/xiachunwei/Software/model_zoo_mindspore/models/official/cv/resnext
export GLOG_v=0 && python3 my_run_resnext101.py > resnext_tmp.txt 2>&1
cat resnext_tmp.txt | grep "xiachunwei-latency"
cat resnext_tmp.txt | grep "xiachunwei-latency"| awk '{ print $8 }'
# 记录一下总的时间和op的数量到表格里

#LSTM
cd /home/xiachunwei/Software/tensor-compiler/src/itvm/operator_fusion/models/lstm/
export GLOG_v=0 && python3 mindspore_lstm.py > lstm_tmp.txt 2>&1
cat resnext_tmp.txt | grep "xiachunwei-latency"| awk '{ print $8 }'
# 记录一下总的时间和op的数量到表格里


#EfficientNet
cd ~/Software/model_zoo_mindspore/models/official/cv/efficientnet
python3 my_run_efficientnet.py --config_path efficientnet_b0_imagenet_config.yaml > efficient_net_tmp.txt 2>&1
cat efficient_net_tmp.txt | grep "xiachunwei-latency"| awk '{ print $8 }'
# 记录一下总的时间和op的数量到表格里

cd /home/xiachunwei/Software/model_zoo_mindspore/models/research/cv/swin_transformer
python3 my_run_swin_transformer.py --swin_config src/configs/swin_base_patch4_window7_224.yaml > swin_transformer_tmp.txt 2>&1
cat swin_transformer_tmp.txt | grep "xiachunwei-latency"| awk '{ print $8 }'
# 记录一下总的时间和op的数量到表格里

# MMOE模型
# 运行MMOE
cd ~/Software/mindspore/model_zoo/official/nlp/bert/src
export GLOG_v=0 && python3 mindspore_mmoe.py > mmoe_tmp.txt 2>&1
# Grep 测量的每个op的时间
cat mmoe_tmp.txt | grep "xiachunwei-latency"
# Grep 出来op的时间
cat mmoe_tmp.txt | grep "xiachunwei-latency"| awk '{ print $8 }'
# 记录一下总的时间和op的数量到表格里


#跑完恢复GPU频率
sudo nvidia-smi -rgc
