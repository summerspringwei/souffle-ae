#!/bin/bash
set -xe

docker exec -it $(docker ps -qf "ancestor=souffle-tensorrt8.4.1-ubuntu18.04:latest") /bin/bash