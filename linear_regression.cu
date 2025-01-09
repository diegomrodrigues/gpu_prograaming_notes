%%writefile "linear_regression.cu"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for linear regression prediction
__global__ void linearRegression(const float* X, const float* beta, float* y, 
                               int n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {  // One thread per sample
        // Initialize prediction with intercept
        float prediction = beta[0];  // β₀
        
        // Add contribution of each feature
        for (int j = 0; j < p-1; j++) {
            prediction += X[idx * (p-1) + j] * beta[j+1];
        }
        
        // Store prediction
        y[idx] = prediction;
    }
}

// Kernel to compute RSS
__global__ void computeRSS(const float* y_true, const float* y_pred, float* rss, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shared_rss[256];  // Assuming max threads per block is 256
    
    float local_rss = 0.0f;
    
    // Each thread computes its local sum of squared residuals
    if (idx < n) {
        float residual = y_true[idx] - y_pred[idx];
        local_rss = residual * residual;
    }
    
    // Store in shared memory
    shared_rss[threadIdx.x] = local_rss;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_rss[threadIdx.x] += shared_rss[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // First thread in block writes result to global memory
    if (threadIdx.x == 0) {
        atomicAdd(rss, shared_rss[0]);
    }
}

int main() {
    // Example parameters
    const int n = 1000;  // Number of samples
    const int p = 4;     // Number of coefficients (including intercept)
    
    // Allocate host memory
    float* h_X = (float*)malloc(n * (p-1) * sizeof(float));  // Features
    float* h_beta = (float*)malloc(p * sizeof(float));       // Coefficients
    float* h_y = (float*)malloc(n * sizeof(float));          // Predictions
    
    // Initialize example data
    // Features X
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p-1; j++) {
            h_X[i * (p-1) + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
    
    // Coefficients β (including intercept β₀)
    for (int j = 0; j < p; j++) {
        h_beta[j] = j + 1;  // Example coefficients: 1, 2, 3, 4
    }
    
    // Allocate device memory
    float *d_X, *d_beta, *d_y;
    cudaMalloc((void**)&d_X, n * (p-1) * sizeof(float));
    cudaMalloc((void**)&d_beta, p * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_X, h_X, n * (p-1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, p * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    linearRegression<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_beta, d_y, n, p);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error launching kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Add synchronization before memory copy
    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print first few predictions
    printf("First 5 predictions:\n");
    for (int i = 0; i < 5; i++) {
        printf("y[%d] = %f\n", i, h_y[i]);
        
        // Print the calculation details
        printf("  Details: %f", h_beta[0]);  // Intercept
        for (int j = 1; j < p; j++) {
            printf(" + %f * %f", h_X[i * (p-1) + (j-1)], h_beta[j]);
        }
        printf("\n");
    }
    
    // Allocate memory for true y values and RSS
    float* h_y_true = (float*)malloc(n * sizeof(float));
    float* h_rss = (float*)malloc(sizeof(float));
    *h_rss = 0.0f;
    
    // Generate true y values using the same model
    for (int i = 0; i < n; i++) {
        h_y_true[i] = h_beta[0];  // Intercept
        for (int j = 1; j < p; j++) {
            h_y_true[i] += h_X[i * (p-1) + (j-1)] * h_beta[j];
        }
        // Add some noise
        h_y_true[i] += ((float)rand() / RAND_MAX) * 0.1f;
    }
    
    // Allocate device memory for true y values and RSS
    float *d_y_true, *d_rss;
    cudaMalloc((void**)&d_y_true, n * sizeof(float));
    cudaMalloc((void**)&d_rss, sizeof(float));
    
    // Initialize d_rss to 0
    float zero = 0.0f;
    cudaMemcpy(d_rss, &zero, sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy data to device
    cudaMemcpy(d_y_true, h_y_true, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute RSS
    computeRSS<<<blocksPerGrid, threadsPerBlock>>>(d_y_true, d_y, d_rss, n);
    
    // Copy RSS back to host
    cudaMemcpy(h_rss, d_rss, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print RSS
    printf("\nResidual Sum of Squares (RSS): %f\n", *h_rss);
    
    // Print first few residuals
    printf("\nFirst 5 residuals:\n");
    for (int i = 0; i < 5; i++) {
        float residual = h_y_true[i] - h_y[i];
        printf("residual[%d] = %f (true: %f, pred: %f)\n", 
               i, residual, h_y_true[i], h_y[i]);
    }
    
    // Clean up
    cudaFree(d_X);
    cudaFree(d_beta);
    cudaFree(d_y);
    cudaFree(d_y_true);
    cudaFree(d_rss);
    free(h_X);
    free(h_beta);
    free(h_y);
    free(h_y_true);
    free(h_rss);
    
    return 0;
} 