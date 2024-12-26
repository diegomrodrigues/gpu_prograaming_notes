**20.1 Background**
*   **Dynamic Workload Variation:** How real-world applications exhibit dynamic variation in the amount of computation required, based on spatial and temporal parameters, showing an example of a turbulence simulation that requires more computation in some areas than others.
*  **Fixed Grid Limitation:** That fixed-grid approaches may waste computational resources by applying fine grids where not necessary, or sacrifice accuracy by using coarse grids where more detail is required, highlighting the limitations of static mapping in dynamic situations.
*   **Dynamic Grid Adaptability:** How dynamic grids can adapt computation to the areas where it’s needed, dynamically adjusting grid resolution to varying levels of detail, thus adapting to the current work, and improving efficiency.
*   **Host-Launched Kernels Limitations:** The necessity of host-launched kernels for all computations in previous CUDA systems, with limited capability for dynamic work discovery and increased burden on host-device communication, as the host needs to initiate all kernel launches.
*   **Dynamic Parallelism Benefit:** How dynamic parallelism enables kernels to launch new kernels directly from the device, reducing host-device communication and burden by enabling on-device management of new workloads.
*   **Dynamic Work Discovery:** Dynamic parallelism capabilities that allow algorithms to discover new work and launch kernels without host intervention, reducing latency and improving performance, and how these new kernels can be launched from other kernels.

**20.2 Dynamic Parallelism Overview**

*   **Kernel-Launched Kernels:** The key concept of dynamic parallelism where a CUDA kernel can launch other kernels, breaking from the host dependency and giving full power to device operations.
*  **Syntax Similarity:** The same syntax for kernel launches in host and device code, with the same parameters and arguments required.
*   **Kernel Launch Parameters:** An overview of the kernel launch parameters like `Dg` (grid dimensions), `Db` (block dimensions), `Ns` (dynamically allocated shared memory), and `S` (stream association), showing the parameters needed for both host and device kernel launches.
*   **Three Level Kernel Launch:**  An example showing a three level hierarchy of kernel launches, showing an entry level kernel launching subsequent level of kernels, highlighting the recursion possibilities.

**20.3 Important Details**

*  **Launch Environment Inheritance:**  That device configuration settings, and device limits, are inherited from parent kernels when launching new kernels, ensuring consistency of execution environments and how the kernel execution environment is controlled by the parent.
*   **Kernel Error Handling:** Recording errors on a per-thread basis in kernels using `cudaGetLastError()`, showing that kernels are still capable of identifying errors through standard mechanisms.
*   **CUDA Events Limitation:** Limitations on using events for intra-stream synchronization, focusing on what is possible in the current CUDA architecture, and how this can affect synchronization options.
*   **`cudaEventCreateWithFlags()` Function:** How dynamic parallelism events should use cudaEventCreateWithFlags(cudaEventDisableTiming) to allow for the correct use of CUDA streams, emphasizing that default event creation is not compatible with dynamic parallelism.
*  **Shared Events:**  That events should be shared only within the threads of a thread block and that it’s usage outside of the allocated block may result in undefined behaviour.
*  **Stream Scope:** Streams created from a kernel are private to the thread block in which they were created, and that passing stream objects between blocks and kernels is not valid.
*   **Stream Concurrency:** That launched work in separate streams may run concurrently, but concurrency is not guaranteed, highlighting that implicit parallelism in different streams can not be guaranteed.
*   **Global Synchronization on NULL Stream:**  That global synchronization using NULL streams from the host are not supported, emphasizing the limitations of using the default stream.
*   **`cudaStreamCreateWithFlags` API Usage:** The requirement to create all streams in the kernel using `cudaStreamCreateWithFlags(cudaStreamNonBlocking)`, ensuring stream safety and proper implementation of dynamic parallelism.
*   **`cudaDeviceSynchronize()` API:**  That in kernel code one can use `cudaDeviceSynchronize()` and that the `cudaStreamSynchronize()` is not supported in dynamic parallelism, with a focus on guaranteeing complete work execution in the device, as there are no other alternatives.
*  **Parent-Child Relationship:** Defining that threads in a parent grid launch child grids and that the parent grid does not complete until all child grids have finished their work, emphasizing a nested execution flow, that ensures that all dependencies between grids are preserved.
*   **Thread Synchronization Scope:** That threads may only synchronize within the scope of the current grid or thread block, using specific synchronization primitives (e.g `cudaDeviceSynchronize`, `_syncthreads`).
*   **Thread Block Scope:**  That streams created by threads inside a thread block will be local to that thread block, and can not be used outside, emphasizing the locality and scoping of created objects.
*  **Host Stream Scope:** The lack of ability to reuse streams allocated in the host inside kernels, as they have undefined behaviour.

**20.4 Memory Visibility**
*  **Coherent Global Memory Access:**  The memory model of parent and child grids accessing global memory coherently, detailing that changes made in global memory in the parent are visible to the child and vice versa, and that synchronization is needed to guarantee correct data values.
*   **Global Memory Operations Visibility:** The point of time when memory operations in a parent become visible to the child (at launch), and how operations of the child are only made visible after the parent synchronizes with the child's termination.
*   **Zero-Copy Memory Coherence:** That zero-copy memory is coherent, similar to global memory, emphasizing that it also follows the same rules and limitations, with the exception that a kernel may not allocate or free zero-copy memory, only use pointers passed from host memory.
*   **Constant Memory Behavior:** How constant memory can not be modified within kernels even in nested launches, highlighting it’s immutability during the entire execution of the dynamic parallelism tree, and that they are globally visible to all kernels.
*  **Constant Memory Address Semantics:** How taking address of a constant object from a thread inside a kernel works the same as a non dynamic parallelism environment, that this pointer can be used as an argument to any other kernel.
*   **Local Memory Scope:** That local memory is private to each thread, and that a thread should not pass pointers to it to child kernels, leading to undefined behaviour if the child tried to dereference that pointer.
*  **Invalid Local Memory Pass:**  An example of passing a local variable to a child kernel, that should be avoided because local variables and registers are not visible to child kernels and it's dereferencing leads to undefined behaviour.
*   **Shared Memory Scope:**  That shared memory is private to a thread block and cannot be used to communicate with other thread blocks, and that it is an error to pass this pointer to a child kernel.
*  **Texture Memory Aliasing:** Texture memory access is read-only, which can be aliased to a global memory region that can be writable, and that texture memory coherence is enforced at the launch and completion of child grids.

**20.5 A Simple Example**

*   **Divergent Phase Emulation:** Emulation of a divergent phase with a simple calculation, and how a work intensive region may cause different threads to execute a different amount of work, with the need for different code paths.
*  **Original CUDA Style Example:** Example of the problem coded using a standard CUDA style kernel, that will cause a lot of warp divergence, and have inefficient execution, even though doing a conceptually simple task.
*   **Atomic Operations in Shared Memory:** The usage of atomic operations to handle concurrent increments of a shared variable, to guarantee correctness and avoiding race conditions.
*   **`threadIdx.x` Check for Execution:** That only some threads in the original kernel will execute the atomic operations, while other threads will finish earlier.
*   **Host Memory Copy and Test:** Copying data back to host and verifying if the output result is what’s expected, highlighting host side operations.
*   **Dynamic Parallelism Example:** Example code in the dynamic parallelism approach, highlighting how threads can launch child kernels to execute work, instead of having the parent thread do the work itself.
*  **`entry()` Kernel Launch:** That the `entry()` function is now a kernel that will be launched from the parent kernel.
*   **Reduced Control Flow Divergence:** How the dynamic parallelism version reduces control flow divergence by launching child kernels that do not perform any divergence or early exits, and showing the improvements over the traditional approach.
*  **Atomic Operations Comparison:**  That even with dynamic parallelism an atomic operation is still required, as these different child kernels may be executed concurrently.

**20.6 Runtime Limitations**
*   **Memory Footprint for Live Threads:**  The large memory footprint associated with dynamic parallelism due to the need to store data of live threads at every level of nesting, and that this memory may be larger than the amount of device memory available.
*   **Nesting Depth Limitation:** The hardware and software limitations on the level of nesting for dynamic parallelism, showing that deeply nested structures are not viable, and that memory usage grows with nesting level.
*    **Memory Allocation and Deallocation**: How `cudaMalloc` and `cudaFree` are modified for use within kernel code, including that cudaFree in host memory should not free device memory and vice versa, and the current memory limits.
*   **ECC Error Notification:**  The lack of notification of ECC errors inside the kernels, and how ECC errors will be reported in the host, emphasizing that kernels will continue to execute with these errors.
*   **Limited Stream Concurrency:**  That despite the unlimited named streams in each block, the maximum concurrency is limited by hardware, and how this may cause serialization or aliasing of streams.
*   **Event Limitations:**  That there is also a limited number of events that are created within blocks, which also consume device memory and may impact concurrency.
*  **Launch Pool Management**: The launch pool is used to track the state of each child kernel, and although it is virtualized, the amount of device memory used by the pool can still be configured.

**20.7 A More Complex Example**

*   **Adaptive Subdivision of Spline Curves:**  Using Bezier curves as a more complex example where adaptive subdivision is used to control the amount of work done, highlighting that dynamic parallelism is useful for variable workloads.
*   **Bezier Curve Definition:** A Bezier curve is defined by a set of control points, and the curves are calculated by linear interpolation of the control points.
*  **Linear Bezier Curve Formula:** The mathematical definition of a linear Bezier curve, it's interpolation formula, and how a line is obtained from two points.
*   **Quadratic Bezier Curve Formula:** The definition of a quadratic Bezier curve, and how it can be seen as a composition of two linear curves, and its formula and parameters.
*  **Bezier Curve Calculation Without Dynamic Parallelism:** Implementation of Bezier curve calculations without dynamic parallelism, with a kernel that computes the coordinates of points in a Bezier curve, calculating a number of points proportional to the curvature of the curve.
*    **Static Allocation of Vertex Storage:** That in the original example the vertex position arrays are statically allocated, leading to possibly unused memory, and having some performance drawbacks, due to divergent behaviour on different data regions.
*   **`computeBezierLine()` Kernel Details:**  Description of the original `computeBezierLine()` kernel, its parameters, thread structure, calculations, and it’s limitations with respect to parallelism, and with potential performance drawbacks.
*   **`computeBezierLinesCDP` Kernel:** Using the `computeBezierLinesCDP()` kernel to launch `computeBezierLinePositions()` kernels, highlighting the two step calculation in dynamic parallelism, first to calculate the work, then to actually do it.
*   **Dynamic Memory Allocation:** That dynamic memory allocation is used within the kernels, reducing the memory footprint and memory usage, only allocating exactly the memory needed by each object, avoiding unnecessary overheads.
*   **Load Balancing:** The dynamic workload adjustment by threads by launching one or more child kernels, and how this is achieved in practice, which leads to improved resource usage through dynamic task distribution.
