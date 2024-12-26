%%writefile "vec_addition.cu"

#include <stdio.h>
#include <stdlib.h>     // for rand() and RAND_MAX
#include <math.h>       // for fabs()
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1000000;  // Vector size
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < N; i++) {
        if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5) {
            printf("Verification failed at %d!\n", i);
            break;
        }
    }
    printf("Vector addition completed successfully!\n");

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
