**13. Parallel Programming and Computational Thinking**

13.1 **Goals of Parallel Computing**
*  Problem Solving Time: Using parallel computing to solve existing problems in less time, with examples such as the financial portfolio risk analysis which can be sped up to provide results faster.
*   Problem Size Expansion: Using parallel computing to solve bigger problems within a given timeframe; using the financial example, if the firm expands the amount of holdings, a sequential version would not be able to finish in time, but the parallel version can.
*   Solution Quality Improvement:  Using parallel computing to achieve better solutions for a given problem and a time restriction, with the example of the risk analysis; using parallel computing, a firm can improve their results while adhering to their given timeframe for results.
*   Motivation for Increased Speed: How parallel computing is primarily motivated by increased speed, and that this speed improvement can be used in several dimensions of the problems: reduce the time for the same problem, reduce time for a bigger problem, reduce time for a more complex model, and also combinations of those different approaches.

13.2  **Problem Decomposition**
*  Parallelism Identification: How parallel applications must be decomposed into subproblems that can be solved concurrently, and this is a process of identification of work to be performed by each unit of parallel execution, where in CUDA is each thread.
*  Threading Arrangement: Consideration of the method used to organize the work among threads, and how these organization are crucial to have good performance, and that two threading arrangements that achieve similar parallel executions can lead to drastically different performances in a given hardware system.
*   Atom-Centric vs. Grid-Centric Arrangement: Two examples of threading arrangements: atom-centric, where each thread calculates the effect of an atom on all grid points, and grid-centric, where each thread calculates the effect of all atoms on one grid point, and how this can lead to different access patterns with different hardware implications.
*  Gather Behavior: The memory access behavior of grid-centric arrangements, where threads gather data to local registers before calculations, and that is generally a desired behavior in CUDA devices due to the possibility of accumulating results in private registers.
*   Scatter Behavior:  The memory access behavior of atom-centric arrangements, where threads scatter data to shared locations in a grid, leading to the necessity of atomic operations and the decrease in performance due to atomic operations and write conflicts.
*  Threading Arrangement Influence: How hardware limitations and thread arrangement limitations dictate to the programmer to seek better gather-based arrangements, to fully use the architecture.
*  Multi-Module Applications: The structure of real applications that often consist of several modules; with the example of the molecular dynamics application where different modules (vibrational, rotational, and non-bonded) might require different execution schemes, and how some modules are better suited for a CUDA device than others.
*  Task-Level Parallelization: The use of task-level parallelization for smaller activities that are not worth running on a CUDA device, and that those tasks can be executed in parallel using multicore hosts or multiple kernels; the use of multicores to execute small tasks in parallel and reduce the limitations imposed by sequential portions of a CUDA program, a concept that is further explored with the usage of MPI.
*   Hierarchical Data Parallelism: Exploitation of data parallelism in hierarchical fashion, by distributing spatial grids in different nodes using MPI, and by using the host CPUs of those nodes for lower-level computations, an approach that allows using all of the computational resources of a complex system.

13.3 **Algorithm Selection**
*   Algorithm Definition: An algorithm as a step-by-step, precisely stated procedure that can be carried out by a computer, and that algorithms should exhibit the three key properties of definiteness, effective computability and finiteness.
*   Algorithm Trade-offs: Multiple algorithms for solving a problem, having distinct characteristics regarding computational complexity, parallel execution level, numerical stability, and memory bandwidth consumption.
*  Tiling Algorithm: How tiling improves memory bandwidth use by loading input data into shared memory before the calculation, and how the introduction of phases implies new synchronizations and overhead, but drastically increases performance by decreasing accesses to the global memory.
*   Merging Threads: How a new algorithm can combine threads to access global memory in an even more efficient manner by merging operations, and using one thread to access the matrix elements that were previously accessed by two threads, thereby reducing the total amount of global memory accesses.
*  Cutoff Binning: The algorithm strategy of cutoff binning and how it improves efficiency by sacrificing a small amount of accuracy, using only local approximations for some calculations.
*   Implicite Methods: How cutoff binning avoids calculating the effects of distant points explicitly, and combines their effects with an approximate method.
*  Direct Summation vs. Cutoff Methods: Comparison of direct summation with cutoff methods for the electrostatic potential problem, in which the cutoff method scales linearly with the volume while the direct method scales quadratically with the volume, and how that causes the sequential version to outperform the GPU version for very small matrixes.
*   Cutoff Algorithm and Scalability: The necessity of different implementation in the parallel environment to efficiently use the algorithm, so that the algorithm does not incur in unnecessary overheads for memory access and synchronization, and that this approach performs well for moderate and large volumes.
*  `LargeBin`, `SmallBin`, and `SmallBin-Overlap`: Three different binned cutoff algorithms, with different characteristics for execution. The LargeBin is simple and performs better for moderate volume systems, SmallBin has higher efficiency due to reduced number of atoms examined by thread, and the SmallBin-Overlap algorithm overlaps the atom processing in host and device to provide slight improvements in running time, reaching a 17x speedup compared to an efficient implementation in CPU using SSE instructions.

13.4 **Computational Thinking**
*   Computational Thinking Definition: The concept of computational thinking is the process of formulating domain problems in terms of computation steps and algorithms and that this concept is a cornerstone for developing better applications, and that this ability comes from bouncing between theory and practice, with real application experiences.
*  Problem Understanding: Requirement of clear understanding of problem to be solved, and a clear grasp of desirable and undesirable memory access behaviors.
*   Overwhelming Algorithm Design: The challenges that are imposed by algorithm design to overcome major difficulties such as parallelism, efficiency and bandwidth consumption.
*   Bottom-Up Learning: Learning through particular models to provide footing before generalizing, making the learning process more natural and easier, since humans learn more efficiently by learning concrete concepts before abstract ones.
*  Essential Parallel Programming Skills:  Summary of skills that the parallel programmer needs to be effective.
    *   Computer Architecture: Memory organization, caching, locality, SIMD, SIMT, SMPD, floating-point accuracy.
     *   Programming Models and Compilers: Memory organization, data layout, thread transformations, loop structure, etc.
    *   Algorithm Techniques: Tiling, cutoff, scatter-gather, and binning.
    *   Domain Knowledge: Numerical methods, precision, accuracy, and stability.

13.5 **Summary**
* Key Aspects Summary: The main dimensions of algorithm selection and computational thinking, and that is necessary to select algorithms with trade-offs.
* Algorithm Variety:  The key lesson that a programmer must select from a variety of different algorithms.
*  Algorithm Tradeoffs: That algorithms present different tradeoffs while maintaining the same numerical accuracy, and that other algorithms might involve sacrificing some accuracy to obtain better results in scaling.
*  Cutoff Strategies: The applicability of cutoff strategies in many different domains that require massive parallelization, and the need for good computational thinking to achieve good results with complex algorithms.

13.6 **Exercises**
*  Implementation of the binning function for atoms, with coalescing and data alignment.
* Implementation of the cutoff kernel.
*  Design of a complete reduction kernel that avoids wasteful threads, with an effective use of configuration parameters.
*   Analysis of the access pattern of threads in the `MatrixMulKernel`, and verification of the access of elements in global memory.
*   Determining input matrices with coalesced access for simple and tiled multiplication.
*  Divergence analysis for reduction kernels, showing the number of warps with divergence.
*   Code analysis of an example using dot product operations.
* Determining block sizes that avoid uncoalesced accesses to global memory.
*  Analysis of a modified kernel to improve performance.

This detailed list should provide a comprehensive learning framework for the topics discussed in this chapter.