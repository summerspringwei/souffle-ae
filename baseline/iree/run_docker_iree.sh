
if [ $# -lt 1 ]; then
    echo "Usage: $0 ["build"|"run"|"attach"]"
    exit 1
fi
# Build docker image
if [ "$1" = "build" ]; then
  docker build -t souffle-iree:latest -f ./iree.Dockerfile .
elif [ "$1" = "run" ]; then
  # Run docker image
  docker run --gpus all -it --privileged\
    -v $(pwd)/iree_models:/workspace/iree_models \
    souffle-iree:latest /bin/bash
elif [ "$1" = "attach" ]; then
  docker exec -it $(docker ps -qf "ancestor=souffle-iree:latest") /bin/bash
fi
