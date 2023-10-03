#!/bin/bash
set -xe

docker pull nnfusion/cuda:10.2-cudnn7-devel-ubuntu18.04

# Define the name of the Docker container you want to check
container_name="souffle-rammer"

# Check if the container is running
if docker ps -q --filter "name=$container_name" | grep -q .; then
    echo "Container $container_name is running. Stopping and removing it..."
    docker stop $container_name  # Stop the running container
    docker rm $container_name    # Remove the container
else
    echo "Container $container_name is not running."
fi

docker run -t --name $container_name \
  -v $(pwd)/rammer_generated_models:/root/rammer_generated_models \
  -d nnfusion/cuda:10.2-cudnn7-devel-ubuntu18.04
docker start $container_name
docker exec -it $container_name /bin/bash
