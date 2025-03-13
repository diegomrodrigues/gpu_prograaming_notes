![Illustration of a 2D convolution operation showing input (N), mask (M), and output (P) arrays.](./images/image1.jpg)

The image, found in Figure 8.4 of Chapter 8, illustrates a 2D convolution example. It depicts input array 'N' as a 7x7 grid of numerical values and 'M' as a 5x5 convolution mask. The result of the convolution operation, 'P', is partially shown, with one element calculated to be 321. The process involves performing pairwise multiplication between elements of the mask array and the corresponding elements of the input array.

![Illustration of 1D convolution: input array N convolved with mask M results in output array P, calculating P[2] as 57.](./images/image2.jpg)

The image, labeled as Figure 8.1 in Chapter 8 of the document, illustrates a 1D convolution example with inside elements. It depicts three arrays: an input array N with seven elements, a convolution mask array M with five elements, and an output array P with seven elements. Arrows show how elements from N and M are used to calculate an element in P, specifically P[2], whose value is 57. The arrays N and P are indexed from 0 to 6, while array M is indexed from 0 to 4, following C language conventions as described in the document. The figure serves to explain the basic concept of 1D convolution, as detailed in section 8.1 on page 174.

![CUDA kernel for 1D convolution, demonstrating parallel computation of output elements.](./images/image3.jpg)

The image presents a CUDA kernel function, `convolution_1D_basic_kernel`, implementing a 1D convolution algorithm as described in Section 8.2 on page 179 of the document. It computes the convolution of an input array `N` with a mask `M`, producing an output array `P`. The code iterates through each element of the output, calculating its value as the weighted sum of neighboring input elements, considering array boundaries using the `Mask_Width` and `Width` parameters.

![Kernel code for tiled 1D convolution, demonstrating shared memory usage and boundary handling (Figure 8.11).](./images/image4.jpg)

The image presents the kernel code for a tiled 1D convolution operation, as referenced in Chapter 8 of the document. The kernel, named `convolution_1D_basic_kernel`, is designed to leverage shared memory for efficient data access. Key steps include loading input data with halo elements into shared memory (`N_ds`), handling boundary conditions by checking for out-of-bounds indices, and performing the convolution using the loaded data. The code uses block and thread indices to map computations across the input array, as discussed in Section 8.4. The `syncthreads()` function ensures proper synchronization after data loading. The image corresponds to Figure 8.11 in the document.

![Simplified diagram of a modern processor's cache hierarchy, showing the levels of cache memory.](./images/image5.jpg)

This image, found on page 184, presents a simplified view of the cache hierarchy in modern processors, emphasizing the proximity and levels of caches to the processor core. It illustrates the relationship between the processor, registers, L1 cache, L2 cache, and main memory, highlighting how data moves between these levels to optimize processing speed. The structure indicates a multi-level cache system designed to reduce latency and improve bandwidth by storing frequently accessed data closer to the processor.

![1D convolution with boundary conditions, showing input array N, mask M, and output array P, where missing elements are padded with zeros.](./images/image6.jpg)

The image illustrates a 1D convolution operation with boundary conditions, corresponding to Figure 8.3 in the document. It depicts arrays N, M, and P, where N is the input array, M is the convolution mask, and P is the output array. Boundary conditions are handled by padding missing N elements with zeros, allowing the convolution operation to be applied even when the mask extends beyond the bounds of the input array, as explained on page 176.

![Illustration of 1D tiled convolution with halo elements, demonstrating input array partitioning.](./images/image7.jpg)

The image illustrates 1D tiled convolution with halo elements as described in Section 8.4. It depicts an input array N divided into four tiles (Tile 0, Tile 1, Tile 2, Tile 3), where each tile represents a thread block processing a subset of N. The diagram shows how halo elements, which are input data elements shared between adjacent tiles, are used to compute the convolution, and ghost elements are represented with dashed boxes.

![Illustration of a 2D convolution boundary condition where missing input elements are treated as zero.](./images/image8.jpg)

The image, referenced as Figure 8.5 in Chapter 8 of the document, illustrates a 2D convolution boundary condition. It depicts a 7x7 input array 'N' being convolved with a 5x5 mask 'M' to produce an output array 'P'. The diagram highlights how, at the boundaries of 'N', the convolution operation requires handling missing elements by assuming a default value, which, in this example, is zero, directly impacting the P values calculation.

![1D convolution showing the application of a mask to an input array N, resulting in output array P with ghost elements for boundary conditions.](./images/image9.jpg)

The image illustrates a 1D convolution operation, where a five-element mask (not explicitly shown but implied by context) is applied to a seven-element input array N to produce a seven-element output array P, as described in Section 8.1 of the document. The image depicts how each element of the output array P is generated by considering a subset of the input array N, including the use of 'ghost' elements (represented by zeros) to handle boundary conditions at the edges of the input array, as referenced in Figure 8.3. This demonstrates a convolution operation with a mask size of 5 and input array size of 7, highlighting the generation of the output array P.

![CUDA memory model: Grid of blocks with shared memory, registers, and threads interacting with global and constant memory.](./images/image10.jpg)

This image, appearing on page 181 of Chapter 8, illustrates a review of the CUDA memory model in the context of constant memory and caching techniques for parallel convolution. It depicts a 'Grid' containing two 'Blocks,' each with 'Shared Memory/L1 cache,' 'Registers,' and 'Threads.' The diagram also shows the 'Host' interacting with 'Global Memory' and 'Constant Memory,' highlighting the data flow and memory hierarchy relevant to optimizing convolution operations.

![1D convolution example showing calculation of P[3] based on input array N and mask M.](./images/image11.jpg)

The image illustrates a 1D convolution operation, as described in Chapter 8 of the document, specifically in Figure 8.2 on page 175. It shows the calculation of the P[3] element, where the 5-element mask 'M' is applied to the input array 'N' to generate the output array 'P'. The calculation involves a weighted sum of neighboring 'N' elements, with the resulting value of 76 being assigned to P[3].
