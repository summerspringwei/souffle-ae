
# Artifact Evaluation

> #### Optimizing Deep Learning Inference via Global Analysis and Tensor Expressions

Optimizing deep neural network (DNN) execution is impor- tant but becomes increasingly difficult as DNN complexity grows. Existing DNN compilers cannot effectively exploit op- timization opportunities across operator boundaries, leaving room for improvement. To address this challenge, we present Souffle, an open-source compiler that optimizes DNN in- ference across operator boundaries. Souffle creates a global tensor dependency graph using tensor expressions, traces data flow and tensor information, and partitions the compu- tation graph into subprograms based on dataflow analysis and resource constraints. Within a subprogram, Souffle per- forms local optimization via semantic-preserving transfor- mations, finds an optimized program schedule, and improves instruction-level parallelism and data reuse. We evaluated Souffle using six representative DNN models on an NVIDIA A100 GPU. Experimental results show that Souffle consis- tently outperforms six state-of-the-art DNN optimizers by 
delivering a geometric mean speedup of up to $3.7\times$ over TensorRT and $7.8\times$ over Tensorflow XLA.

## Preliminaries
This repository showcases the performance evaluation and comparison between `Souffle`(Our work) and the existing state-of-the-art compilers/frameworks(including `XLA`, `TensorRT`, `Rammer`, `Apollo` and `IREE`).

- **XLA(Tensorflow v2.10)**: The TensorFlow XLA compiler
can fuse DNN operators like point-wise and reduction op-
erators and performs optimizations on the fused operator.
Unlike Souffle that performs analysis and optimizations on
TEs, XLA performs analysis on its high-level operators(HLO)

- **TensorRT(v8.2)**:This GPU-vendor-specific framework optimizes the inference of DNNs on NVIDIA GPUs

- **Rammer(v0.4)**:This DNN compiler is also known as NNFu-
sion. It generates a spatial-temporal schedule at compile
time to minimize scheduling overhead and exploit hardware
parallelism through inter- and intra-operator co-scheduling

- **Apollo**:This represents the state-of-the-art fusion framework for inference optimization. Apollo considers both
memory- and compute-bound tensor operators for kernel
fusions and uses hand-crafted rules to exploit parallelism
between independent tensor operators

- **IREE**:The intermediate resentation execution environment (IREE) builds upon the
LLVM MLIR project. IREE is designed to lower DNN
models to MLIR dialects to optimize model inference. IREE
utilizes the linalg dialect to perform the operator fusion,
which supports loop affine fusion optimization and global
analysis.

The metric assesed in this notebook mainly include end-to-end latency, global memory access and number of kernels.

## Important Notes
**A few bash scripts take more than half an hour to complete; Please wait for the results before executing the next one.**

# For models with python bindings, please first run `pip install .` in the corresponding directory.
docker run -it --name=sirius_test --gpus all --privileged sunqianqi/sirius:mlsys_ae /bin/bash


### Links to The Paper

**For each step, we highlight that the current evaluation is corresponding to which Section or Figure in the submitted paper.**
The main restuls presented in this repository correspond to the submitted paper's Table 3, 4, 5 and Figure 6.

## 1. Experimental Environments Setup

### 1.1 Requirements

#### Hardware：
An linux server with 
* an NVIDIA A100 Tensor Core GPU, 
* at least 128GB memory and 
* an Intel CPU with at least 8 physical cores.
* High-bandwidth network access is also required as we need to download about 50 GB software and docker images to build the experimental environments.

#### Software:
* Operating System: Ubuntu 18.04
* Docker version >= 20.10.14
* CUDA 11.7 and compatible drivers

### 1.2 Build all env
We provide a script to build all the env in just one commond:
```shell
bash scripts/build_all_docker.sh
```
For a linux server with 20MB/s Ethernet network and a Intel Xeon CPU with 20 core, it will takes about three hours to build all the dockers.
The actual building time mainly depends on the network bandwidth and the number of CPU cores.

### 1.3 Start all the dockers
Run the following commond the start the dockers:
```shell
bash run_tvm_0.8.sh run
cd baseline/iree && bash run_docker_iree.sh
```

### 1.4 Check the status of docker containers
```shell
docker ps
```
You should see containers with the following image names:
* souffle-tvm-0.8:latest
* souffle-iree:latest
* souffle-tensorrt8.4.1-ubuntu18.04:latest

## 2. Evaluation
Next, we use four cases to test the end-to-end performance for the baseline compilers and our work.
Each case matches a table or figure in the submitted paper.

we recommend you to run these cases one by one, which reduces the total execution time at most **two hours**.
We also provided a fast mode which re-use the existing profiling data 
to directly print the outputs.

- 2.1 CASE - End-to-end model runtime (Table 3 in Section 8) - around 30 minutes.
- 2.2 CASE - Execution time with Souffle individual optimization (Table 4 in Section 8) - around 30 minutes.
- 2.3 CASE - The number of GPU kernel calls and global memory
data transfer size of the resulting code (Table 5 in Section 8) - around 50 minutes.
- 2.4 CASE - EfficientNet sub-module latency breakdown (Figure 6 in Section 8) - around 10 minutes.

**Log files**

PS: some cases would consume over a half-hour because we have to execute all baselines. Please have a coffee and wait for the output before the subsequent execution.

### 2.1 CASE - End-to-end model runtime (Table 3)
In this case, we compare souffle with five representative state-of-the-art baselines to exploit the end-to-end latency.
We omit Ansor in this case.

| Model       | XLA  | TRT   | Rammer | Apollo | IREE  | Ours |
| ----        | ---- | ----  | ----   | ----   | ----  | ---- |
| BERT        | 2.55 | 1.30  | 2.19   | 3.29   | 2.22  | 1.22 |
| ResNeXt     | 8.91 | 24.82 | 11.69  | 22.80  | 314.8 | 4.43 |
| LSTM        | 10.57| 6.30  | 1.72   | Failed | 16.0  | 0.80 |
| EfficientNet| 2.96 | 1.21  | Falied | 2.3    | 12.33 | 0.66 |
| SwinTrans.  | 6.43 | 1.74  | Falied | 10.78  | 18.1  | 1.55 |
| MMoE        | 0.29 | 0.07  | Falied | 0.049  | 0.088 | 0.014|

**The following commonds reproduce the results of Table 3 in the submiited paper. Please refer to Section8(Page 10) for more details**

Run the experiments:
```shell
bash scripts/run_table3.sh
```
Check for the results:
```shell
cat results/table3.csv
```
Note that the results in `table3.csv` is a transposed matrix of table 3 in the submmited paper.