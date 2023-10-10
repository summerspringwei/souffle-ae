# Apollo

## Build
This experiments requires the `tvm_0.8` docker and use the same env.
The framework is built during the buiding of the docker.

## Run 
First run `tvm_0.8` docker at root of `souffle-ae`:
```shell
bash run_tvm_0.8.sh run
```
Then execute the following commonds to get the execution latency of all the models except `LSTM`.
```shell
cd /wordspace/baseline/mindspore
bash run_nsys_apollo.sh
``` 
