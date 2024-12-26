**3.1 Data Parallelism**

*   **Data Parallelism in Modern Applications:** The inherent data parallelism present in applications processing large datasets, specifically those modeling real-world phenomena like images, physics simulations, and scheduling problems.
*   **Independent Evaluation as Basis:**  The principle that independent computations on different data elements form the core of data parallelism.
*   **Task Parallelism vs. Data Parallelism:**  Distinction between decomposing applications into independent tasks versus operating on independent data elements concurrently.
*   **Task Decomposition:** The methodology of dividing an application into discrete, independent units of work.
*   **Scalability of Data Parallelism:** Data parallelism as the primary driver for achieving scalability in parallel programs, leveraging increasing hardware resources.
*   **Role of Task Parallelism in Performance:** The supplementary role of task parallelism in optimizing performance, particularly in conjunction with data parallelism.
*   **Vector Addition Example of Data Parallelism:**  Illustrative example where each element-wise addition in a vector operation represents an independent parallel operation.

**3.2 CUDA Program Structure**

*   **Host-Device Model in CUDA:** The fundamental architectural separation between the host (CPU) and one or more devices (GPUs) in a CUDA environment.
*   **Mixed Host and Device Code:** The ability to integrate both CPU and GPU code within a single CUDA source file.
*   **Default Host Code:**  The interpretation of standard C code within a CUDA program as code intended for CPU execution.
*   **CUDA Keywords for Device Constructs:** The use of specific keywords to identify functions and data structures intended for GPU execution.
*   **NVCC Compilation Process:** The role of the NVIDIA CUDA Compiler (NVCC) in separating and compiling host and device code.
*   **Host Code Compilation Flow:** Compilation of host code using standard C/C++ compilers and execution as a traditional CPU process.
*   **Device Code Compilation Flow:**  Marking of data-parallel functions (kernels) and data structures using CUDA keywords, and subsequent compilation by the NVCC runtime.
*   **Kernel Functions:** The fundamental units of parallel execution on the CUDA device, encapsulating data-parallel operations.
*   **Threads in Modern Computing:** Conceptual understanding of a thread as a unit of program execution within a processor.
*   **CUDA Kernel Launch and Thread Generation:**  The mechanism by which a CUDA program initiates parallel execution by launching kernel functions, leading to the creation of numerous threads.
*   **Efficiency of CUDA Thread Management:** The hardware-level optimizations that enable rapid generation and scheduling of CUDA threads compared to traditional CPU threads.
*   **Grids of Threads:** The collective term for all threads launched by a single kernel invocation.
*   **Kernel Execution Lifecycle:** The sequence of host CPU execution initiating kernel launches, parallel GPU execution of threads, and the eventual termination of the grid, followed by continued host execution.
*   **Overlapping CPU and GPU Execution:** Advanced techniques for concurrently executing code on the CPU and GPU to maximize resource utilization.

**3.3 A Vector Addition Kernel**

*   **Conventional CPU-Only Vector Addition:**  Review of a standard sequential C function for vector addition, highlighting its iterative nature.
*   **Host and Device Variable Naming Conventions:**  The use of prefixes (e.g., `h_`, `d_`) to differentiate variables primarily processed by the host or device, respectively.
*   **Memory Allocation and Initialization in Host Code:** The process of allocating and populating input and output vectors in the CPU's memory.
*   **Passing Array Pointers to Functions:** The mechanism of passing array data to functions in C using pointers, enabling access and modification.
*   **Revised `vecAdd` Function for CUDA:**  Structural modification of the vector addition function to offload computation to a CUDA device.
*   **CUDA Header Inclusion:** The necessity of including `<cuda.h>` to access CUDA API functions and built-in variables.
*   **Device Memory Allocation:**  Allocating memory on the GPU's dedicated memory space to store data for parallel processing.
*   **Host-to-Device Data Transfer:**  Copying input data from the host's memory to the allocated device memory.
*   **Kernel Launch for Parallel Execution:**  Initiating the execution of the vector addition kernel on the CUDA device.
*   **Device-to-Host Data Transfer:**  Copying the computed results from the device memory back to the host memory.
*   **Outsourcing Computation to the Device:** The concept of the revised `vecAdd` function acting as an intermediary, delegating the computationally intensive task to the GPU.

**3.4 Device Global Memory and Data Transfer**

*   **Separate Memory Spaces in CUDA:** The architectural distinction between the host's system memory and the device's dedicated global memory (DRAM).
*   **Global Memory as Device Memory:**  Interchangeable terminology for the GPU's primary memory space.
*   **Necessity of Explicit Data Transfer:** The requirement to explicitly move data between host and device memory for kernel execution.
*   **CUDA Runtime API for Memory Management:** The set of functions provided by CUDA to handle memory allocation, deallocation, and data transfer.
*   **`cudaMalloc()` for Device Memory Allocation:**  API function for allocating memory within the device's global memory space.
*   **Similarity to Standard C `malloc()`:**  Conceptual and functional parallels between `cudaMalloc()` and the standard C memory allocation function.
*   **`cudaFree()` for Device Memory Deallocation:** API function for releasing allocated memory within the device's global memory.
*   **Pointer Usage with Device Memory:**  The concept of pointers referencing memory locations on the GPU and the restrictions on dereferencing device pointers in host code.
*   **`cudaMemcpy()` for Data Transfer:**  API function for copying data between host and device memory, or between different locations within the same memory space.
*   **Parameters of `cudaMemcpy()`:** Understanding the roles of the destination pointer, source pointer, number of bytes to copy, and transfer type.
*   **Transfer Types in `cudaMemcpy()`:**  Specifying the direction of data transfer (host to device, device to host, host to host, device to device).
*   **CUDA Error Handling Mechanisms:** The importance of checking return flags from CUDA API functions to detect and manage errors during execution.

**3.5 Kernel Functions and Threading**

*   **CUDA Kernel Functions:** Functions designated for execution by multiple threads in parallel on the CUDA device.
*   **SPMD (Single Program, Multiple Data) Paradigm:** The underlying programming model where all CUDA threads execute the same kernel code on different portions of the data.
*   **Grids and Thread Blocks:** The hierarchical organization of CUDA threads into grids of thread blocks.
*   **Uniform Block Size Within a Grid:** The constraint that all thread blocks within a single grid have the same dimensions.
*   **Maximum Threads Per Block:** The hardware-imposed limit on the number of threads within a single thread block.
*   **`blockDim` Variable:**  A built-in variable accessible within a kernel that provides the dimensions of the current thread block.
*   **`threadIdx` Variable:** A built-in variable providing a unique identifier for each thread within its thread block.
*   **Unique Global Thread ID Calculation:** The method for computing a unique index for each thread across the entire grid using `blockIdx` and `threadIdx`.
*   **`blockIdx` Variable:** A built-in variable providing the index of the current thread block within the grid.
*   **Hardware Efficiency Considerations for Block Dimensions:** The performance implications of choosing thread block dimensions that are multiples of a specific value (often 32).
*   **CUDA Function Qualifiers:**  The use of `__global__`, `__device__`, and `__host__` keywords to specify the execution context and callability of functions.
*   **`__global__` Keyword:** Designating a function as a kernel, executable on the device and callable from the host.
*   **`__device__` Keyword:** Designating a function for execution on the device, callable only from other device functions or kernels.
*   **`__host__` Keyword:** Designating a function for execution on the host (CPU), callable only from other host functions.
*   **Combined `__host__` and `__device__`:**  Enabling compilation of a single function for both host and device execution.
*   **Implicit Loop Replacement with Thread Grid:** The concept that the parallel execution of threads within a grid effectively replaces explicit loop structures for data parallelism.
*   **Conditional Execution Within Kernels:**  The use of conditional statements (e.g., `if (i < n)`) within kernels to handle cases where the data size is not a multiple of the number of threads.
*   **Kernel Launch Configuration Parameters:** The syntax `<<<grid_dim, block_dim>>>` for specifying the dimensions of the grid and thread blocks when launching a kernel.
*   **Ceiling Function for Grid Dimension Calculation:**  Using the ceiling function to ensure sufficient thread blocks are launched to cover the entire dataset.


