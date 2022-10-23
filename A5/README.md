
# Configure
Only if you have CUDA 8 or newer, when you enabled to use extended lambada feature. [New Compiler Features in CUDA 8](https://developer.nvidia.com/blog/new-compiler-features-cuda-8/)




# Problem Record

When I use `--expt-extended-lambda` option, linker seems to be abnormal.
```shell
/usr/bin/ld: cannot find -lcudadevrt
/usr/bin/ld: cannot find -lcudart_static
```

["/usr/bin/ld: cannot find -lcudart"](https://askubuntu.com/questions/510176/usr-bin-ld-cannot-find-lcudart) in `ask ubuntu` forum successfully solved this problem.
```shell
sudo ln -s /usr/local/cuda/lib64/libcudart_static.a /usr/lib/libcudart_static.a
sudo ln -s /usr/local/cuda/lib64/libcudart_static.a /usr/lib/libcudadevrt.a
```

> [1] [Thinking Parallel, Part I: Collision Detection on the GPU](https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/)
> [2] [Thinking Parallel, Part II: Tree Traversal on the GPU](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/)
> [3] [Thinking Parallel, Part III: Tree Construction on the GPU](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)
> [Resource] [Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees)
> [Resource] [tinyobjloader-v1.0.6](https://github.com/tinyobjloader/tinyobjloader/tree/v1.0.6)