#!/bin/bash
set -xe
script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

container_name="souffle-rammer"

# Check if the Docker image exists
if docker image inspect "$container_name" &>/dev/null; then
    echo "The Docker image $container_name exists."
else
    echo "The Docker image $container_name does not exist."
    docker pull sunqianqi/sirius:mlsys_ae
fi
