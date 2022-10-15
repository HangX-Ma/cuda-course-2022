# 第四次作业要求

1. 内容：实现基于流的计算加速：如何使用stream/pinned memory实现计算和内存传输的overlap，对比性能提升。

2. 细节：基于我提供的`vectorAdd.cu`进行修改，对比不使用stream，两个stream，三个stream的加速效果。

3. 参考：`cuda by example`第十章的内容和例子，对比下书中两种使用两个stream的方法对于性能的影响。

4. 工具：visual profiler，分析stream的加速原因。