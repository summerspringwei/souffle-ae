#!/bin/bash
set -xe

# docker pull sunqianqi/sirius:mlsys_ae

# Define the name of the Docker container you want to check
container_name="souffle-rammer"

# Check if the container is running
if docker ps --filter "name=${container_name}" | grep -q .; then
    echo "Container ${container_name} is running. Stopping and removing it..."
    docker stop ${container_name}  # Stop the running container
    docker rm ${container_name}    # Remove the container
else
    echo "Container ${container_name} is not running."
fi

# Run the docker in backgroud
docker run -td --name ${container_name} \
  -v $(pwd)/rammer_generated_models:/root/rammer_generated_models \
  --gpus all --privileged \
  sunqianqi/sirius:mlsys_ae /bin/bash

# First build all the models
docker exec -it ${container_name} /bin/bash /root/rammer_generated_models/build_rammer.sh
# Then run all the models
docker exec -it ${container_name} /bin/bash /root/rammer_generated_models/run_rammer.sh
docker stop ${container_name}
