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


Dear reviewers:

**Thank you for your efforts. Please see the following comments:**

## Response for comments A12

**response for Table3**
1. Time for ResNeXt on XLA is 10x bigger
> We forget the filter out the memory copy/boundary checker (`redzone_checker`) / layout convert kernels produced by XLA. These kernels are auxiliary kernels and are not for operator computation.
We have fixed the bug and provided a stanalone cell the check the results for XLA on ResNext and EfficientNet.
See `http://8.141.164.33/notebooks/asplos24ae.ipynb#Standalone-cell-for-unexpected-result`

**response for Table5**
1. TRT on LSTM memory transfer and # of kernel calls is bigger;
> After examining our original setup, we find that we set the wrong configuration for the LSTM in the submitted paper:
The number of layers should be 10 (AE version) rather than 8 (submitted paper version).
And the number produced in the AE is correct. We will fix the numbers in the final version.
As you have mentioned the conclusion still stands. 
The results are beneficial for our work as Souffle produces fewer kernels compared with the baseline compilers.

2. XLA number of kernels on ResNeXt is 20x bigger,
>  We have fixed the bug and provided a stanalone cell the check the results for XLA on ResNext.

3. XLA on Efficient is 10x bigger
> The same as ResNeXt.
We have fixed the bug and provided a standalone cell the check the results for EfficientNet.

4. The author's method on Swin-Trans is 4x bigger.
> We will check our code to fix the bug and provide a standalone Jupyter cell to reproduce the result.

PS. The results for XLA are not stable even thought we use the same version of TensorFlow and same code.
The XLA always runs unexpected kernels (like `redzone_checker`).
Even though we filter out some kernels the number of kernels still much larger than that of ours.

**response for Figure6**
1.  fused method in M3,M4 and AVG has lower speed up*
> The speedup variance for for M3, M4 and AVG is small (around $10\%$).
The average fused method speed-up and global-sync speed-up for AE are 1.48 and 1.55, respectively.
The speed up in the submitted paper is 1.49 and 1.40.
So the speed up is a little higher than that of the submitted paper.

Thank you so much for your patience and All the best!
