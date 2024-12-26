6.1 **Warps and Thread Execution**

*   Thread Execution Order: Independence of thread execution order within a block, emphasizing the need for explicit synchronization; threads should not rely on any implicit ordering.
*   Warp-Based Execution: Bundling of threads into warps for SIMD hardware execution, and that the hardware executes all threads in a warp in a single instruction multiple data manner.
*   SIMD Implementation: Low cost of SIMD hardware, with its ability to execute a single instruction for multiple data, leading to lower manufacturing costs, and also the possibility for power savings.
*   Divergent Control Flow: Definition of thread divergence; where threads within a warp execute different instructions, causing the hardware to take multiple passes to account for all control flow paths, this multipass approach is what allows to preserve the independence of threads while leveraging the SIMD nature of the execution units.
*  Cost of Divergence: Additional execution passes are required due to different paths, and that these additional passes directly increase the overall execution time.
*   Causes of Divergence: Use of control constructs (like `if-else` and loops) with conditions based on `threadIdx`, a pattern that causes thread divergence and can impact performance.
*   Reduction Algorithm: How reductions can be implemented using parallel execution, but also that this example of a highly utilized algorithm naturally presents thread divergence.
*   Parallel Sum Reduction: How the reduction algorithm derives a single value from an array of data, and that this operation can be done in parallel using multiple threads.
*   Sequential Reduction Algorithm: The nature of a sequential implementation of reduction algorithms, which is work efficient, but that is inherently slow due to its serial nature.
*   Work Efficiency vs. Hardware Utilization: Difference between work efficiency of algorithms and efficient utilization of hardware execution units; even if an algorithm does minimal work, it can be slow if it doesn’t allow an efficient parallel execution.
*   Initial Sum Reduction Kernel: Highlighting the thread divergence in the initial implementation of the sum reduction kernel, due to the conditional statement based on `threadIdx`.
*   Revised Sum Reduction Kernel: How the revised version has less thread divergence, by using operations that are more coherent in the beginning, and with a small divergence only at the very end of the algorithm.
*   Loop Iterations and Divergence Reduction: Understanding how the modification to the reduction algorithm enables for less divergence, reducing the overhead of divergent control flow.

6.2 **Global Memory Bandwidth**
*   Global Memory Access: Importance of global memory access, and the typical need to perform massive data transfers from global memory.
*  Memory Coalescing: Memory coalescing techniques, that allows CUDA devices to reach peak performance; hardware optimization in DRAM systems that combines multiple requests from threads into a single access, and how this speeds up memory access if done in the right way.
*   DRAM Memory Storage: DRAM storage of data using capacitors, and that reading requires the capacitor to drive a high capacitance line, with a slow detection mechanism.
*  DRAM Parallelism: How parallelism is used in DRAMs to increase the data access rate; accessing consecutive memory locations and having the data available at a high rate.
*   Favorable Access Patterns: How threads in a warp accessing consecutive locations result in hardware coalescing, this pattern of memory access allows the hardware to deliver data close to the peak bandwidth.
*  Unfavorable Access Patterns:  How random or non-consecutive memory accesses inhibit coalescing, and how that reduces the effective throughput of memory, and the need to use proper access patterns for optimal performance.
*   Row-Major Layout: Explanation of row-major memory layout for multi-dimensional arrays, where elements in a row are stored consecutively in memory.
*  Coalesced Access Examples: The case of matrix multiplication where threads in a warp access a column from the matrix `N`, to demonstrate coalesced memory accesses.
*   Uncoalesced Access Examples: The case of matrix multiplication where threads in a warp access a row from the matrix `M`, to demonstrate uncoalesced memory accesses.
*   Tiling and Shared Memory for Coalescing: How shared memory is used to transform global memory accesses that are not coalesced in the global memory to coalesced accesses within the shared memory.

6.3  **Dynamic Partitioning of Execution Resources**
*   SM Resources: List of execution resources in SM, such as registers, shared memory, thread block slots, and thread slots, and that these resources are partitioned dynamically between thread blocks and assigned to threads to support their execution.
*  Dynamic Partitioning: The ability of SM to dynamically partition threads slots and register files between blocks, and how this partitioning varies depending on thread and block sizes, and the code needs.
*   Thread Slots and Blocks Slots: Dynamic partitioning of thread slots and how that allows for versatile use of SM, by supporting many small or few big thread blocks, and that a fixed assignment of resources leads to waste if blocks does not fully utilize the provided resources.
*   Register Allocation and Thread Count:  The relationship between register usage and the number of thread blocks, and how the shared memory per block also affects the maximum amount of blocks that can be scheduled in a SM.
*   Interaction of Resource Limitations: The fact that resources limitations, such as shared memory or registers, can interact with each other, leading to underutilization of resources, as seen with the number of thread slots being wasted due to lack of registers or memory.

6.4 **Instruction Mix and Thread Granularity**
* Instruction Processing Bandwidth: Each SM has a limited instruction processing bandwidth; every instruction consumes a portion of the bandwidth.
*   Redundant Work and Bandwidth: How redundancy between threads affects the instruction processing bandwidth, especially in code with redundant computations.
*  Thread Granularity: The concept of granularity of threads, and how is advantageous to put more work in each thread, and use less threads, when some work is redundant between threads.
*   Redundant Loading Example: The matrix multiplication example, with the redundant loading of `d_M` matrix tiles into shared memory, how the new approach will reduce the redundancy.
*  Adjusting Granularity for Efficiency: The opportunity to adjust the thread granularity to reduce redundancy, and by having each thread do more work and that this can help to reduce the overall cost of performing a highly parallel computation.
*   Rectangular Tiles: How rectangular tiles improve performance with larger matrices, even if that reduces the overall amount of thread blocks, it increases performance due to less access to the global memory.
