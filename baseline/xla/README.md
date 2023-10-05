## Build
This experiments requires the `tvm_0.8` docker and use the same env.

## Run
First run `tvm_0.8` docker at root of `souffle-ae`:
```shell
bash run_tvm_0.8.sh run
```

then run:
```shell
cd baseline/xla
docker exec -it $(docker ps -qf "ancestor=souffle-tvm-0.8:latest") /bin/bash /workspace/baseline/xla/run_xla.sh
```

## Note
For `swin-transformer`, only part of the operators are supported by XLA, so we only optimize the MLP with XLA.
