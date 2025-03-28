![Memory allocation and deallocation behavior of `cudaMalloc()` and `cudaFree()` from host and device.](./images/image1.jpg)

Table 20.1 from page 448 of the document describes the behavior of `cudaMalloc()` and `cudaFree()` when used on the host and device, detailing which operations are supported in each environment, in the context of CUDA dynamic parallelism. Specifically, it indicates that `cudaFree()` can only free memory allocated by `cudaMalloc()` in the same environment, and the allocation limit differs between host and device.

![Illustration comparing fixed versus dynamic grids for turbulence simulation, demonstrating adaptive mesh refinement for performance optimization.](./images/image2.jpg)

This image, labeled as Figure 20.1 on page 436, illustrates the difference between fixed and dynamic grids in the context of a turbulence simulation model within a CUDA environment; the "Initial Grid" demonstrates a baseline representation, while the "Fixed Grid" statically assigns a uniform grid, regardless of the level of detail needed, and the "Dynamic Grid" dynamically adjusts the grid refinement based on the accuracy requirements in different areas, enabling performance optimization; arrows visually connect the initial state to the subsequent grid configurations, highlighting the transition and adaptation processes.

![Parent-child kernel launch nesting demonstrating CUDA dynamic parallelism execution flow.](./images/image3.jpg)

This image, found on page 7 of the document, illustrates the concept of Parent-Child Launch Nesting in CUDA dynamic parallelism. It depicts a CPU thread launching a 'Grid A' kernel, which then launches a 'Grid B' kernel; this demonstrates kernel launch from within another kernel and highlights how parent and child kernels execute in sequence according to the given timeline. Synchronization between parent and child kernels ensures proper execution order and data visibility.

![Illustration of kernel nesting in CUDA dynamic parallelism, where kernel B launches child kernels X, Y, and Z.](./images/image4.jpg)

The image, found in Figure 20.3 on page 438, illustrates a basic example of CUDA dynamic parallelism, contrasting it with the traditional CUDA model where kernel launches occur from the host code; the main function (host code) initially launches kernels A, B, and C, and then kernel B subsequently launches kernels X, Y, and Z, a functionality not permitted in prior CUDA implementations. It demonstrates the nesting of kernels, a key feature of dynamic parallelism, where a kernel can spawn other kernels, managed within the device.

![Comparison of kernel launch patterns: (a) without dynamic parallelism and (b) with dynamic parallelism.](./images/image5.jpg)

This image, Figure 20.2 from page 437, illustrates a conceptual comparison between CUDA with and without dynamic parallelism. On the left, the diagram shows a traditional CUDA implementation where the CPU launches a series of kernels, each represented as a rectangle on the GPU; the CPU then receives information from these kernels and launches the next set. On the right, the diagram illustrates dynamic parallelism, where threads within a kernel on the GPU can launch additional kernels without CPU intervention, represented as a hierarchical tree of rectangles.

![Valid and invalid examples of passing pointers to child kernels in CUDA dynamic parallelism (Figure 20.5 from page 443).](./images/image6.jpg)

The image, found on page 443 of the document, illustrates valid and invalid scenarios for passing pointers as arguments to a child kernel in CUDA dynamic parallelism. The valid example (a) shows 'value' defined in global storage using the `__device__` specifier, allowing its address to be passed. In contrast, the invalid example (b) defines 'value' within the local scope of function y(), making it illegal to pass its address to a child kernel, which would result in undefined behavior.
