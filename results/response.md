Dear reviewers:
Many thanks for your time and efforts.

**Response for comments on table 3**
1. The unit for Souffle is `microsecond (us)` while the unit for baselines is `millisecond (ms)`.
We have fixed the inconsistency. Thank you for pointing out.

4. The abnormal value of SwinTrans. on XLA is also that we use the wrong units as before.

2. The end-to-end latency for Souffle is around 10% longer than the latency reported in the paper.
The reason is that the results in the submitted paper are measured by repeatedly running 10000 times and then computing the average number.
to save time for AE, they are measured by using nsight-compute and **only run once without warmup**.

3. Some results for IREE are different than those in the submitted paper.
The reason is that IREE and its dependencies update every day.
We failed to rebuild the corresponding version of IREE in the submitted paper.
and we use the latest version in the AE.

PS. The main purpose of this table is to compare the end-to-end latency of our work and STOA works.
Even though some numbers do not fit the values in the submitted paper well (e.g. for XLA and IREE),
the conclusion still holds: our work can significantly reduce the end-to-end latency compared with competing baselines.

**Response for comments on table 4**
1. We have fixed the abnormal value of ResNext with the V0 version.

2. We have fixed the path error and it now can produce the corresponding results for SwinTransformer and MMoE.

3. The execution time for EfficientNet (V1) should be 4.2 rather than 0.91 in the submitted paper.
We will fix the error in the final version.

PS. The main purpose of this table is to demonstrate that our work can reduce the latency by enabling our optimization step-by-step.


**Response for comments on table 5**
1. Lack the results for XLA
> I have added the number of kernels on XLA in the `run_table5.sh`.
What's more, I also added a standalone cell to only run the XLA so that you can save time.
2. TRT's data is a little bit larger
> We forgot to filter out the memory copy kernel in resnext before and now we have fixed this issue.
The number of kernels is reduced from 3588 to 2421,
and the memory access is reduced from 822 MB to 754 MB.

PS. The main purpose of this table is to demonstrate that our work can significantly reduce the number of kernels and memory access.

**Response for comments on figure 6**
1. higher\lower relationship for **M3,M4 and AVG**
> In terms of the higher\lower relationship, 
The relative speed-up between `fused` and `global-sync` could be different from the figure in the submitted
paper as these are two independent optimizations.
The speed up for `fused` may be higher or lower than that of `global-sync`.
Secondly, to shorten the running time we only run the kernel once to profile the latency,
so the number may be slightly different from the number in the submitted paper.
