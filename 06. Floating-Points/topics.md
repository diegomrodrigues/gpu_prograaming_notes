**6.1 Warps and Thread Execution**

*   **CUDA Thread Hierarchy:** Review of the two-level thread hierarchy of grids and blocks, emphasizing their role in organizing parallel computations within CUDA.
*   **Transparent Scalability:** The concept of blocks executing independently, providing transparent scalability in CUDA kernels, allowing for easy scaling of the workload.
*  **SIMD Hardware and Warps:** The concept of Single Instruction, Multiple Data (SIMD) execution, where the same control signal is applied to multiple processing units that execute a warp of threads simultaneously.
*   **Control Unit and Instruction Fetch:** The role of a single control unit in fetching and decoding instructions for multiple processing units, highlighting the architectural design that affects instruction execution, and how they can reduce costs.
*   **Thread Synchronization:** Emphasis on using barrier synchronization to ensure all threads complete a phase of execution, before proceeding to the next phase, critical for correct parallel computation.
*  **Warp Partitioning:** How CUDA devices group threads into warps for execution, reducing hardware manufacturing costs and optimizing memory access patterns.
*   **Warp Size:** The standard size of a warp in CUDA devices (32 threads), showing how threads are bundled together for execution.
*   **Thread Indexing and Warp Mapping:** The process of partitioning thread blocks into warps based on thread indices (e.g., threadIdx.x) and how dimensions are projected into linear order for warp assignment.
*   **Thread Divergence:** The concept of threads within a warp following different execution paths due to conditional statements, leading to performance penalties, such as the SIMD hardware executing all paths sequentially.
*   **Divergence Cost:** The extra hardware passes needed to allow threads in a warp to take their own control flow path, and thus impacting performance.
*   **Control Flow Divergence and `threadIdx`:** Demonstrating how conditional statements based on `threadIdx` values can cause thread divergence, and how loops may also lead to divergence, impacting performance.
*   **Reduction Algorithm as Example:** The use of a reduction algorithm to exemplify divergence within thread execution, highlighting that using loop condition based on thread index values can cause divergence.
*   **Parallel Reduction Algorithm:**  The structure of the parallel reduction algorithm that resembles a soccer tournament, using multiple rounds of pairwise reductions to achieve a single output, and how it motivates parallel execution.
*  **In-place Reduction:** How the shared memory array is used to store the input array and is then used to store the reduction results, by replacing partial sums in the array.
*   **Barrier Synchronization in Reduction:** The importance of `syncthreads()` to ensure all partial sums are generated before the next reduction step, highlighting synchronization needs.
*   **Thread Selection in Reduction:** How the if statement is used to select the even threads to participate in the reduction operation in a loop, and how different steps result in a smaller set of threads participating.
*   **Stride Variable:** How the stride variable defines the distance between the elements added in the reduction steps, and how the stride changes in each step.
*   **Work Efficiency vs. Hardware Utilization:** Balancing the work-efficiency of an algorithm against hardware utilization considerations, and the need to effectively utilize available hardware.
*   **Modified Sum Reduction Kernel:** Using a revised sum reduction algorithm to reduce thread divergence by using a stride that is initially half the size of the block, improving efficiency.
*    **Thread Positions and Divergence:** How the positions of threads that execute the `add` statement influences performance, comparing it to the original code with thread divergence.

**6.2 Global Memory Bandwidth**

*   **Global Memory Access:** The importance of global memory bandwidth for CUDA kernel performance, due to the massively parallel nature of applications and how they tend to use a massive amount of data.
*   **Memory Coalescing:** Techniques for improving data transfer rates from global memory to shared memory and registers, emphasizing memory bandwidth optimizations.
*  **DRAM Implementation:** How the global memory in CUDA devices is implemented with DRAMs that uses capacitors to store data, detailing the hardware that influences read and write latency.
*   **DRAM Access Latency:** The slow process of reading data from DRAM cells relative to desired access speeds, emphasizing the reasons for using parallelization.
*   **Parallelism in DRAMs:** How modern DRAM chips use parallelism to increase the data access rate, highlighting the importance of accessing multiple locations at once.
*   **Consecutive Access and Bandwidth:** The increase in bandwidth if accesses are focused on consecutive memory locations, and how data is transferred from the sensor at very high speed.
*  **CUDA Memory Access Patterns:** How CUDA devices optimize for global memory access, by detecting if the threads in a warp access consecutive global memory locations to combine them into a single request for data.
*   **Coalesced Accesses:** Combining multiple memory accesses into a single request to consecutive memory locations, to improve global memory bandwidth.
*   **Data Layout and Row-Major Convention:** The standard row-major layout of multi-dimensional arrays in C and CUDA, where elements of a row are placed into consecutive locations in memory.
*  **Row Major Access Patterns:** How the placement of data in memory impacts data access patterns for 2D arrays.
*   **Uncoalesced Access:** How accessing data in a row-major manner with different threads reading in a column-wise manner causes uncoalesced access patterns, leading to poor performance.
*   **Tiled Algorithms for Coalescing:** How shared memory is used to load tiles into a coalesced pattern, then accessed later in a different manner, showcasing the combination of shared memory and coalesced accesses.
*   **Matrix Multiplication Coalescing Example:** The use of a matrix multiplication kernel example to illustrate favorable vs. unfavorable access patterns for coalescing, highlighting the data access patterns in the kernel code.
*   **Shared Memory for Coalescing:** Using shared memory in tiled algorithms to enable coalesced access to global memory, loading the data into a contiguous memory space and later using it in a different format.
*   **Data Access Pattern Analysis:** Detailed examination of the memory access patterns for both d_M and d_N matrices in matrix multiplication to understand which matrix have a coalesced access.
*   **Consecutive Thread Accesses:** The importance of consecutive thread indices for accessing consecutive memory locations, and how the hardware detect them and combine them into a coalesced access.

**6.3 Dynamic Partitioning of Execution Resources**

*   **Streaming Multiprocessor (SM) Resources:** How the SMs have multiple resources, such as registers, shared memory, thread block slots, and thread slots to optimize resource allocation.
*   **Dynamic Resource Partitioning:** The concept of dynamically assigning resources like registers, shared memory, thread blocks, and thread slots based on a program's needs, allowing better hardware utilization.
*   **Thread Block Slots and Dynamic Allocation:** How the 1,536 thread slots are dynamically partitioned and assigned to thread blocks during runtime, focusing on hardware resource utilization.
*  **Thread Block Resource Allocation:** How resources are allocated dynamically to better utilize hardware and how SMs can handle different numbers of thread blocks at once.
*   **Fixed vs. Dynamic Partitioning:** Comparing dynamic partitioning with fixed partitioning, showing how fixed partitioning can waste resources, and how dynamic partition gives more flexibility.
*   **Resource Limitation Interactions:** The potential for interactions between different resource limitations causing underutilization, such as using block and thread slots together.
*   **Register Allocation and Its Effects:** The impact of declaring automatic variables on register usage, illustrating dynamic register allocation and how this influences the number of blocks that can run on each SM.
*   **Performance Cliff:** The sudden drop in performance when resource limitations are exceeded, showcasing the importance of understanding resource usage.
*  **CUDA Occupancy Calculator:** The use of the CUDA Occupancy Calculator (NVIDIA) to determine the actual number of threads running on each SM, given resource usage by the kernel, showing an important tool for optimizing resource usage.

**6.4 Instruction Mix and Thread Granularity**

*   **Thread Granularity Tuning:** The algorithmic decision to adjust the amount of work done per thread to improve performance, balancing work per thread and the overall number of threads.
*   **Redundant Work Between Threads:** How redundant computations between threads can hinder performance, and how such redundancies can be eliminated by increasing granularity.
*   **Instruction Processing Bandwidth:** The limitation imposed by the instruction processing bandwidth of the SM, highlighting that every instruction uses instruction bandwidth.
*   **Reducing Redundant Instructions:** Eliminating redundant instructions to ease pressure on instruction processing bandwidth and improve overall kernel speed.
*   **Matrix Multiplication Redundancy:** A matrix multiplication example to illustrate redundant loads of the same d_M row by multiple thread blocks, showing how a given tiled matrix algorithm can cause redundant reads.
*   **Merging Thread Blocks and Redundancy:** How the merging of adjacent thread blocks reduces redundant loads and how a single thread can handle computations that were previously performed by more than one thread.
*  **Adjusted Kernel for Higher Granularity:** The process of adjusting kernels to compute more than one d_P element per thread, and how the dot product is computed, emphasizing optimization through increasing granularity.
*   **Increased Registers and Shared Memory:** Potential downsides of merging thread blocks, like increased register and shared memory usage, impacting the number of blocks running on each SM.
*   **Parallelism and Reduced Thread Blocks:** How reducing the number of thread blocks might negatively impact parallelism for smaller matrices, highlighting potential performance trade-offs, emphasizing the importance of proper use cases.
*   **Horizontal Block Combining:** The concept of combining horizontal blocks to improve performance in large matrix multiplication, and how using it may improve performance of large matrices (2048x2048 or more).
