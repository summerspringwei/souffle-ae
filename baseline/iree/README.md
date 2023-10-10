## Build
We have provided a docker file to build the iree env from scratch.
Run the following commond to build the docker:
```shell
bash run_docker_iree.sh build
```

## Run models
Run the following commands in the docker:
```shell
cd iree_models
bash run_nsys_iree.sh
```
The scripts will run all the six models