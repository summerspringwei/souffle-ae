## Build
This experiments requires the `tvm_0.8` docker and use the same env.

## Run
First run `tvm_0.8` docker and then run
```shell
docker exec -it $(docker ps -qf "ancestor=tvm-0.8:latest") /bin/bash run_xla.sh
```

## XLA model files

101 machine: `/home2/xiachunwei/Software/fusion/xla_models`