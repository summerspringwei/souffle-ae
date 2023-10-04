#!/bin/bash
set -xe

if [ $# -lt 1 ]; then
    echo "Usage: $0 ["build"|"run"|"attach"]"
    exit 1
fi

# Build docker image
if [ "$1" = "build" ]; then
  docker build -t souffle-rammer:latest -f ./rammer.Dockerfile .
elif [ "$1" = "run" ]; then
  # Run docker image
  docker run --gpus all -it --privileged\
    -v $(pwd)/rammer_generated_models:/workspace/rammer_generated_models \
      souffle-rammer:latest /bin/bash
elif [ "$1" = "attach" ]; then
  docker exec -it $(docker ps -qf "ancestor=souffle-rammer:latest") /bin/bash
fi
