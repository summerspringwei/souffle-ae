#!/bin/bash
set -xe

script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

container_name="souffle-rammer"

# Define the name of the Docker container you want to check
# Check if the container is running
if [ "$1" = "rm" ]; then
if docker ps --filter "name=${container_name}" | grep -q .; then
    echo "Container ${container_name} is running. Stopping and removing it..."
    docker stop ${container_name}  # Stop the running container
    docker rm ${container_name}    # Remove the container
else
    echo "Container ${container_name} is not running."
fi
fi

# if there is'nt the container, create it
if [ ! "$(docker ps -a --filter "name=souffle-rammer")" ]; then
  docker run -td --name ${container_name} \
    -v ${script_directory}/rammer_generated_models:/root/rammer_generated_models \
    --gpus all --privileged \
    sunqianqi/sirius:mlsys_ae /bin/bash
  docker exec -it ${container_name} /bin/bash /root/rammer_generated_models/build_rammer.sh
# else, if the container is not running, start it
elif [ ! "$(docker ps -qf "name=souffle-rammer")" ]; then
    docker start ${container_name}
fi
# else, do nothing, the container is already running
# Then run all the models
docker exec -it ${container_name} /bin/bash /root/rammer_generated_models/run_rammer.sh
