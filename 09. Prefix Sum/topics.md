**9.1 Background**

*   **Inclusive Scan Operation:** Definition of an inclusive scan (prefix sum) operation, which returns an array where each element is the cumulative sum of the original array up to that point, using a binary associative operator, emphasizing that the last value is the full sum of all elements.
*   **Exclusive Scan Operation:** How exclusive scans are similar to inclusive scans, differing that they output the cumulative value up to the previous element in the array, thus the first element is zero, with the final result not including the last element, and it's importance in memory allocation.
*  **Conversion between Inclusive and Exclusive Scans:** How to convert the output between inclusive and exclusive scans using a shift and fill operation, emphasizing the relationship between both.
*   **Applications of Parallel Scan:** The applications of parallel scan in diverse algorithms, such as radix sort, quicksort, string comparison, and many other computations, highlighting the pattern's versatility.
*   **Sequential Scan Algorithm:** A work-efficient sequential implementation of an inclusive prefix sum algorithm, and it's characteristics, that takes only one addition, one load and one store for each element, and thus being very efficient.

**9.2 A Simple Parallel Scan**

*   **Reduction Tree Approach:**  Using a reduction tree for calculating prefix sums for each output element in parallel, to achieve a high level of parallelism.
*   **In-Place Scan Algorithm:** The concept of performing scan operations in-place using a shared memory array `XY`, where values are updated directly, while also using it as output.
*  **Visualizing the Algorithm:** The use of a diagram to illustrate the iterative calculation of the parallel scan algorithm using reduction tree, step-by-step, showing the values updated in the shared memory.
*  **`XY` array evolution:**  How the  `XY` array evolves in each step of the reduction tree operation, and that the element in the `XY` array will contain the sum of 2n previous inputs at iteration n.
*    **Thread Assignment:** The assignment of a thread to each position of `XY`, showing how the calculation is organized in parallel.
*    **Kernel Implementation:** Implementation of the parallel scan in a CUDA kernel, with each thread calculating the sum of the previous elements, using the concept of a reduction tree.
*   **`SECTION_SIZE` Constant:** The importance of `SECTION_SIZE` constant, which represents the size of the data handled by each block, that also defines the block size and represents the amount of shared memory used by the kernel.
*  **Loop Implementation:** How the reduction tree algorithm is implemented as a loop in the kernel, accumulating values with each iteration, with control flow divergence on the first warp.
*   **Barrier Synchronization:**  The need of using `_syncthreads()` to synchronize thread calculations at each step of the reduction tree, guaranteeing data integrity and correct calculations.
*   **Control Flow Divergence:** The presence of a modest level of control flow divergence due to smaller thread indices exiting the loop early and how this divergence is reduced with large block sizes.
*   **Exclusive Scan Conversion:** The easy way to convert the inclusive scan kernel to an exclusive scan kernel by shifting the elements and adding 0 to the first position of the output, showing a small change that can lead to a different result.

**9.3 Work Efficiency Considerations**

*   **Algorithm Complexity Analysis:** The analysis of the work complexity of the parallel scan kernel, emphasizing that each thread will perform up to `log2(N)` steps and that the number of threads that do not perform the summation is defined by the stride.
*   **Total Number of Add Operations:** Calculation of the total number of addition operations done by all threads in the parallel kernel as *N log₂(N) - (N - 1)*, showing that it is much larger than in the sequential method.
*   **Work Efficiency Comparison:** Comparing the number of additions performed by the parallel scan vs. the sequential scan for several section sizes, and that the parallel algorithm does a lot more work than the sequential algorithm.
*    **Hardware Resource Requirements:** That to break even the performance with the parallel algorithm one needs a large amount of parallel hardware, and that because of the high number of operations and energy consumption that the parallel algorithm is not suited for power-constrained devices.
*   **Inefficiency of Simple Parallel Scan:**  That the simple algorithm is not a scalable solution due to the high number of operations performed, thus showing the importance of using a more optimized algorithm.

**9.4 A Work-Efficient Parallel Scan**

*   **Opportunities for Sharing:** Identifying opportunities for sharing intermediate results to optimize performance, using the reduction tree, showing that certain sums calculated in the first step can be used by other results later.
*   **Reduction Tree for Sums:** The use of a reduction tree for generating sum values for a set of values in a log2(N) steps, with the reduction tree being responsible for distributing the partial sums in the data set, and also generating subsums.
*   **Partial Sum Generation:**  Using the reduction tree to generate several subsums for each position in the `XY` array, highlighting that each element in the array holds different levels of partial sums.
*   **Reverse Tree Distribution:** The use of a reverse tree to distribute the partial sums to the correct positions, showing that these generated values will be then used to produce the scan results.
*    **Addition Patterns in Tree:** A detailed description of addition patterns in reduction and distribution trees, where different sets of elements are updated in each stage, with varying distances between values summed.
*   **Divergence Elimination:** How the revised reduction tree algorithm has less divergence among threads in a warp, by making use of consecutive threads.
*   **Tree Implementation:** Implementation of the reduction and distribution tree through loops using consecutive threads, highlighting the usage of the stride variable to control the thread access pattern, and also highlighting the usage of syncthreads to correctly synchronize the values.
*   **Work-Efficient Kernel:** The final implementation of a work efficient inclusive kernel, which avoids unnecessary operation while also providing a good performance gain.
*   **Number of Add Operations:** That the work-efficient kernel performs about 2N additions, showing that the new version is much more work-efficient than the previous version.
*    **Number of Threads:** That the kernel only uses N/2 threads for both the reduction and distribution phases.

**9.5 Parallel Scan for Arbitrary-Length Inputs**

*   **Hierarchical Approach:** Addressing the challenge of processing large datasets by partitioning them into smaller sections, showing the scalability of the algorithm.
*   **Multi-Block Processing:** How a large input can be partitioned across multiple thread blocks, each handling a section, in a divide and conquer like method.
*  **Two-Level Hierarchy Implementation** That the two-level hierarchy will need three kernels for the implementation.
*  **First Level Kernels:** How the first kernels are responsible for the scan on each section of the data, highlighting that the output of each scan block are only partial results, and only represents the sum of the elements in that specific block.
*   **Second Level Kernel:** How a second kernel is used to compute the scan results for last values of all blocks of data and the third kernel for distributing these values to the first level results, in order to produce the final values for each item.
*  **Final Result Composition:** That the final results are computed by adding the scan results from both the first and second kernel on all individual positions, showing a correct composition of results of different hierarchical levels, and finally resulting in the full prefix sum.
*   **Carry-Lookahead Adder Analogy:** Similarity between the hierarchical scan algorithm and the carry-lookahead method used in computer arithmetic, highlighting its performance and parallel efficiency.
*   **Native Exclusive Scan Kernel:** That the algorithm can easily be converted to an exclusive version by altering the way the data is loaded into the array and also by reading the paper Harris 2007.
