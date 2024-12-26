**19.1 Background**

*   **GPU Adoption in Supercomputers:** The increasing adoption of GPUs in top supercomputers for energy efficiency, due to the performance per watt ratio, a very relevant criteria for high-performance computing.
*   **MPI as Dominating Interface:**  That the Message Passing Interface (MPI) is the dominating programming interface for clusters, highlighting its importance as an interface for distributed computing.
*   **Distributed Memory Model in MPI:** MPI's assumption of a distributed memory model, where processes communicate by exchanging messages, without direct memory access between different processes, emphasizing its usage in clusters.
*  **MPI API and Communication Details:** That MPI abstracts the details of the interconnect network from the user, allowing portability between different systems by using a logical rank system.
*   **Data and Work Partitioning:** How data and work are divided among MPI processes in a typical application, with each process responsible for a given subset of the workload and data.
*   **Inter-process Data Needs:** How processes need to exchange data with each other to complete a larger task, emphasizing the communication challenge.
*   **Collective Communication APIs:**  The usage of collective communication APIs for synchronization and collaboration, showing how a common work step between processes is coordinated.

**19.2 A Running Example**

*   **3D Stencil Computation:**  A 3D stencil computation for heat transfer using finite difference method is used as a running example, introducing a real-world application that will be used throughout the chapter.
*   **Higher-Order Stencil:** How to compute each point using a weighted sum of its neighbor's values in a higher-order stencil computation, with 4 points in each direction, totalling 24 neighbors and 25 points in the computation, emphasizing the spatial locality of the algorithm.
*    **Data Representation:** How the data is represented in a structured grid, using a 3D array and spacing variables to describe the system, and how this translates into the code.
*  **Domain Partitioning:** Dividing the 3D array into domain partitions, each to be processed by a different MPI process, to allow work distribution in the cluster.
*   **Z Slice Memory Location:**  That elements from the same Z slice are in contiguous memory locations, simplifying memory accesses during computation and data transfers.

**19.3 MPI Basics**

*   **SPMD Execution Model:**  How MPI programs are based on the Single Program, Multiple Data (SPMD) model, where the same program executes on all processes, and how this impacts program design.
*   **`MPI_Init()` Function:**  Initialization of the MPI runtime by each process using `MPI_Init()`, highlighting the setup and allocation of system resources for all processes in the computation.
*   **`MPI_Comm_rank()` Function:**  Retrieving a unique process ID (rank) within a communicator using `MPI_Comm_rank()`, and how the ranks of processes are analogous to thread indexes in CUDA.
*   **MPI Communicator (`MPI_Comm`):**  The usage of the MPI communicator to specify the scope of the request, and that processes are identified inside an intercommunicator.
*   **`MPI_COMM_WORLD` Communicator:** How to use `MPI_COMM_WORLD` to establish communication among all processes running an application, showing an important communicator for collective operations.
*   **`MPI_Comm_size()` Function:** Obtaining the total number of MPI processes using `MPI_Comm_size()`, so that each process knows its position in the whole computation.
*   **Error Checking in MPI:** How MPI programs should check for a sufficient number of processes, using `MPI_Comm_abort()` to terminate in case of insufficient processes, or any other issue, thus allowing robust programming.
*  **Data Server Process:** The separation of roles into compute processes and a data server process for data initialization, and it’s use for IO operations.
*   **`MPI_Finalize()` Function:** Freeing MPI communication resources and ending application execution with `MPI_Finalize()`, showing that resources are reclaimed at the end of the program.

**19.4 MPI Point-to-Point Communication Types**

*   **Point-to-Point Communication:** A description of point-to-point communication, which involves communication between a single source and destination process, illustrating how direct communication between processes is done.
*   **`MPI_Send()` Function:** The process of sending data using `MPI_Send()`, highlighting data transfer to a specified process in the system, with information about the data to be sent, destination rank, and communicator.
*   **`MPI_Recv()` Function:** Receiving data from a specified process using `MPI_Recv()`, highlighting data reception from a specified process, with information about the data to be received, sender rank, and communicator.
*   **`MPI_Datatype`:** How data types are specified using `MPI_Datatype`, which includes built-in data types, such as integers, floats, and doubles, and it’s importance in communication compatibility.
*   **MPI Tag:**  The role of message tags in `MPI_Send()` and `MPI_Recv()`, used to classify messages sent by the same source, showing how multiple send operations may be identified in a single process.
*   **`MPI_ANY_TAG` Constant:** The use of `MPI_ANY_TAG` to specify that the destination process is willing to accept messages of any tag value from a source, for situations in which the tag is not relevant.
*   **Data Server in MPI Example:**  How data server is used to initialize the data with random numbers, and distribute to compute processes, showing a way to abstract the complexity of IO in a cluster system, and how this reduces the example's complexity.
*  **Data server code** How to specify the number of processes, compute the number of points and bytes to be sent, the memory allocation for the data, how to calculate the starting address of the data, and how the `MPI_Send` function is used to distribute data.
*  **Compute Process Code:**  How to obtain the number of process and compute the number of points and bytes to be sent, and how to receive data from the data server, also accounting for halo slices in the memory allocation.

**19.5 Overlapping Computation and Communication**

*   **Inefficient Compute-Communicate Cycles:**  The issue of sequential computation and communication patterns, forcing processes to either compute or communicate, with periods where the hardware is not well utilized.
*   **Two-Stage Approach:** Using a two-stage approach to overlap computation and communication, with one stage dedicated to computation that prepares the data for communication, and another that exchanges the data while also computing on the internal points.
*   **Boundary Slice Computation:** That each compute process will first compute the boundary slices for the next iteration, and how it is important to compute these first to enable the data to be sent to the corresponding nodes in a parallel fashion, improving communication efficiency.
*   **Data Exchange Implementation:** Implementation of data exchange through host memory staging using async copies, and `MPI_Sendrecv` function, allowing computation to run in parallel to communication.
*   **Pinned Memory Buffers:** The use of pinned (page-locked) memory buffers to ensure that the memory is not paged out during DMA operations, guaranteeing memory access reliability and performance.
*   **`cudaHostAlloc()` Function:** How CUDA uses `cudaHostAlloc()` to create pinned memory buffers, while standard `malloc()` function does not.
*   **Memory Paging:** What paging is and how it affects DMA memory operations and how the operating system manages the memory, and why pinning the memory can avoid data corruption.
*   **`cudaMemcpyAsync()` Function:** Using asynchronous copies with `cudaMemcpyAsync()` for efficient data transfer between host and device memory, enabling parallel computation and communication.
*    **CUDA Streams:** The usage of CUDA streams to allow concurrent execution of CUDA API operations, allowing for better performance through parallelism, with a single operation per stream, but multiple concurrent streams.
*  **`MPI_Sendrecv()` Function:**  The combined send and receive operations using `MPI_Sendrecv()`, which is used in compute processes to enable them to send and receive data in a single API call, simplifying the communication code.
*    **Data Exchange Logic:** That the data exchange is implemented with conditional assignements to prevent unnecessary calls to `MPI_Sendrecv()`, and how edge nodes will use MPI_PROC_NULL.
*   **Synchronization for Device Activities:**  The synchronization of device activities using `cudaDeviceSynchronize()` to ensure data transfer and computations are complete before proceeding to the next stage, and how it will enforce a specific execution order.
*   **Swapping Input and Output Pointers:** Swapping the input and output data pointers to be used in the next iteration, with the use of a conditional statement for the last computation step, and emphasizing that a new operation starts from where the previous one ended.

**19.6 MPI Collective Communication**

*   **Collective Communication:** Introduction to collective communication, which involves a group of MPI processes, in contrast to point-to-point communication, that only communicates between two nodes.
*   **`MPI_Barrier()` Function:** The usage of `MPI_Barrier()` as an example of collective communication, for synchronizing processes before data exchange and before starting new compute iterations, and how it guarantees that all nodes will reach a specific point before proceeding.
*   **Other Collective Communication Types:** Mention of other collective communication types, such as broadcast, reduction, gather, and scatter and it’s importance for collaboration among multiple processes.
*  **Optimization of Collective Communication:** How these collective communication functions are highly optimized, and how they are encouraged to be used for better performance, reliability, and readbility.
