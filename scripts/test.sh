#!/bin/bash
set -x
script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! "$(docker ps -qf "ancestor=sunqianqi/sirius:mlsys_ae")" ]; then
bash ${script_directory}/../baseline/rammer/run_docker_rammer.sh start
fi
bash ${script_directory}/../baseline/rammer/run_docker_rammer.sh run