
if [ $# -lt 1 ]; then
    echo "Usage: $0 ["build"|"run"|"attach"]"
    exit 1
fi
# Build docker image
if [ "$1" = "build" ]; then
  docker build -t tvm-0.8:latest -f ./tvm_0.8.Dockerfile .
elif [ "$1" = "run" ]; then
  # Run docker image
  docker run --gpus all -it --privileged\
    -v /home/xiachunwei/Software/tensor-compiler:/workspace/tensor-compiler \
    -v /home/xiachunwei/Projects/tensor-compiler-gpu/:/workspace/tensor-compiler-gpu \
    -v /home/xiachunwei/Projects/bert_rammer:/workspace/bert_rammer \
    -v /home/xiachunwei/Projects/Swin-Transformer:/workspace/Swin-Transformer \
    -v /home2/xiachunwei/Software/fusion/xla_models:/workspace/xla_models \
    tvm-0.8:latest /bin/bash
elif [ "$1" = "attach" ]; then
  docker exec -it $(docker ps -qf "ancestor=tvm-0.8:latest") /bin/bash
fi
