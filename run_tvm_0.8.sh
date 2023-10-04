
if [ $# -lt 1 ]; then
    echo "Usage: $0 ["build"|"run"|"attach"]"
    exit 1
fi
echo $(pwd)/baseline/mindspore
# Build docker image
if [ "$1" = "build" ]; then
  docker build -t souffle-tvm-0.8:latest -f ./docker/tvm_0.8.Dockerfile ./docker
elif [ "$1" = "run" ]; then
  # Run docker image
  docker run --gpus all -it --privileged\
    -v $(pwd)/souffle-models:/workspace/souffle-models \
    -v $(pwd)/baseline/:/workspace/baseline/ \
      tvm-0.8:latest /bin/bash
elif [ "$1" = "attach" ]; then
  docker exec -it $(docker ps -qf "ancestor=tvm-0.8:latest") /bin/bash
fi
