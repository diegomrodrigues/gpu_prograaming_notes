**10.1 Background**

*   **Sparse Matrix Definition:** The fundamental concept of sparse matrices, where the majority of elements are zero, contrasting them with dense matrices, focusing on their prevalence in various modeling problems.
*   **Sparse Matrices in Linear Systems:** The use of sparse matrices to represent coefficients in linear systems, illustrating how each row maps to an equation and the sparsity of real world systems.
*   **Compressed Sparse Row (CSR) Format:** A description of the CSR format, which is commonly used to store sparse matrices by storing only the non-zero values, along with the corresponding column indices and row pointers, highlighting memory savings.
*   **`data[]` Array:** A 1D array used to store all the nonzero values in the sparse matrix sequentially, illustrating how non zero data is stored in the CSR format.
*   **`col_index[]` Array:** The array storing column indices of each element in the `data` array, highlighting the importance of this array for correct matrix access and computations.
*   **`row_ptr[]` Array:** An array containing pointers/indices to the beginning locations of each row's non-zero elements in the `data` and `col_index` arrays, crucial for accessing specific rows in the CSR representation and it’s usage in variable row lengths in sparse matrices.
*    **Boundary row_ptr:** How the last index of row_ptr array is used to define the ending position of the last row in the sparse matrix.
*  **Iterative Solution Approach:** Use of iterative methods like conjugate gradients to solve sparse linear systems, highlighting alternatives to direct matrix inversion, and emphasizing their use in large and sparse systems.
*   **SpMV as Time-Consuming Step:** Highlighting the importance of efficient sparse matrix-vector multiplication (SpMV) in iterative solvers, emphasizing that the most time-consuming part of iterative methods is the multiplication and accumulation operations.
*   **Dense Vectors in SpMV:** That in SpMV, typically the vectors x and y are dense, i.e. most of their values are non zero values, showing how these vectors complement the sparse matrix A.
*   **Standardized SpMV Interfaces:** The creation of standardized library interfaces for SpMV, facilitating the reuse of high performance algorithms for iterative methods.
*   **Trade-offs in SpMV:**  Introduction to the performance trade-offs between storage formats and computation methods in sparse matrices that will be highlighted with the use of SpMV.

**10.2 Parallel SpMV Using CSR**

*   **Independent Dot Product Calculations:** The inherent parallelism in calculating dot products for each row of a sparse matrix, as each operation is independent from one another.
*   **Mapping Rows to Threads:**  How each row of the matrix is assigned to a separate thread for parallel computation, mapping work to individual threads for efficient parallel processing.
*   **Parallel SpMV/CSR Kernel:**  A simple CUDA kernel that uses each thread to perform a dot product for a specific row, illustrating the simplicity of implementation based on independence between rows.
*   **Thread Index Calculation:** The usage of `blockIdx.x`, `blockDim.x`, and `threadIdx.x` to calculate the appropriate row index, a commonly used pattern in CUDA.
*    **Handling Arbitrary Number of Rows:** The use of a conditional statement in the kernel to handle the case where the number of rows is not a multiple of the thread block size, showing correct handling of different data shapes.
*   **Non-coalesced Memory Accesses:** The problem of non-coalesced memory accesses by adjacent threads in the SpMV/CSR kernel, due to simultaneous access to non adjacent memory locations, making the kernel inefficient.
*   **Control Flow Divergence in SpMV/CSR:** Potential control flow divergence due to varying numbers of non-zero elements in different rows causing different amount of work among the different threads, with varying levels of control flow divergence in each warp.
*   **Data Dependent Performance:** That performance in SpMV/CSR heavily depends on the input data, highlighting that different matrices can cause different levels of performance, unlike previous examples of compute-intensive examples.

**10.3 Padding and Transposition**

*  **Non-coalesced Access and Divergence Solutions:** The use of data padding and transposition to address the problems of non-coalesced memory accesses and control flow divergence in SpMV.
*  **ELL Storage Format:** Introduction to the ELL format from the ELLPACK sparse matrix package by padding rows to the maximum number of non-zero elements, highlighting memory access and divergence improvements.
*   **Padding Zero Elements:** Adding dummy (zero) elements to shorter rows to make all rows have the same length, to make the matrix a rectangular matrix, enabling coalesced memory access and lower divergence.
*  **Padded `col_index` Array:**  That the col_index array is also padded to preserve its correspondence to padded data values, showing that all the components in CSR format will have to be padded.
*   **Column-Major Layout:** How data is laid out in column-major order after padding the data, which effectively transposes the data, with all elements of the first column placed together in memory, then the second and so on.
*   **`row_ptr` Elimination:** How after transposition, the `row_ptr` array is no longer needed, since the number of elements are the same across all rows, which allows for more efficient indexing.
*   **Index Calculation Simplification:**  That after transposition, accessing next element in the row is as simple as adding the number of rows to the current index, showing a performance improvement due to simplified indexing.
*  **Simplified SpMV/ELL Kernel:** A CUDA kernel using the ELL format, emphasizing its simplicity due to padded data structure, which is now free from divergence and non coalesced accesses.
*   **Equal Iterations in Dot Product Loop:** That all threads in a warp now iterate exactly the same number of times in the dot product loop, thanks to the padding step, eliminating control flow divergence among the threads in a warp.
*   **Coalesced Memory Accesses in SpMV/ELL:** How the data access pattern in SpMV/ELL enables coalesced memory accesses, which is achieved due to data padding and transposition, thus leading to better memory performance.
*   **Limitations of ELL:** Potential inefficiencies when some rows have an extremely large number of non-zero elements, leading to excessive padded elements that will need to be computed but will not impact the final result, highlighting that SpMV/ELL might have a performance downfall if some rows are too long.

**10.4 Using Hybrid to Control Padding**

*   **COO Format:** Introduction to the coordinate (COO) format as a solution for controlling excessive padding in ELL, detailing the storage of each non-zero element alongside its row and column index.
*    **`row_index` and `col_index` Arrays:** That COO format uses row_index and col_index arrays to store information about the location of each element of data[], emphasizing the flexibility of the format, allowing any order in the data array.
*   **Flexibility of COO Format:** Arbitrary reordering of COO elements and its impact on flexibility, and how the location of each element is determined through the index arrays.
*   **COO for Reducing Padding:** Using COO to store long rows and using ELL to represent short rows, limiting unnecessary memory usage by using two different formats.
*   **Hybrid ELL-COO Method:**  How hybrid approach can combine ELL and COO formats for SpMV, storing elements of short rows using ELL and storing long rows in COO format.
*    **SpMV in ELL:** Performance of SpMV in ELL format using the padded data.
*   **SpMV in COO** Performance of SpMV in COO using atomic operations.
*   **Host-Device Interaction:** A typical hybrid ELL-COO usage, where the host handles COO parts and the device processes the ELL representation, showcasing how CPU and GPU can cooperate in a heterogeneous environment.
*    **Amortized Cost** How iterative methods can amortize the cost of converting the matrix to ELL/COO hybrid form, as iterative methods will perform the SpMV multiple times in the same sparse matrix.
*   **Atomic Operations in SpMV/COO:** How multiple threads performing accumulation on the `y` array requires atomic operations to avoid race conditions when using COO format, due to the non specific mapping between rows and threads.

**10.5 Sorting and Partitioning for Regularization**

*   **Sorting Rows by Length:** Sorting rows based on the number of non-zero elements to reduce padding overhead, leading to a more regular representation of the data.
*   **Jagged Diagonal Storage (JDS) Format:** The concept of jagged diagonal storage (JDS) format, used for sparse matrices after row sorting, and how it often looks like a triangle-shaped structure, and it’s connection to triangular matrix storage.
*   **`jds_row_index` Array:** The importance of the `jds_row_index` array to keep track of the original row positions of the sorted matrix.
*   **JDS-ELL Representation:** How JDS can be combined with ELL format by dividing the matrix into sections, sorting rows within sections, and padding each section to the maximal length within that section, highlighting the trade offs between regularization and memory overhead.
*   **Transposition in JDS-CSR:** How memory coalescing relaxes alignment requirements in more modern devices and that JDS-CSR can be transposed directly, avoiding padded elements.
*   **Performance of JDS Formats:**  That while JDS-ELL provides good performance in older devices, JDS-CSR tends to give better performance in more recent devices, especially with Fermi and Kepler architectures, demonstrating performance impact of device architecture.
