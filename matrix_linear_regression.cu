%%writefile matrix_linear_regression.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel para calcular X*β (predições)
__global__ void computeXBeta(const float* X, const float* beta, float* Xbeta, 
                           int n, int p) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n) {
        float prediction = beta[0];  // Intercepto
        for (int j = 0; j < p-1; j++) {
            prediction += X[row * (p-1) + j] * beta[j+1];
        }
        Xbeta[row] = prediction;
    }
}

// Kernel para calcular (y - Xβ) (resíduos)
__global__ void computeResiduals(const float* y, const float* Xbeta, 
                               float* residuals, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        residuals[idx] = y[idx] - Xbeta[idx];
    }
}

// Kernel para calcular (y - Xβ)ᵀ(y - Xβ) (RSS)
__global__ void computeRSS(const float* residuals, float* rss, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shared_rss[256];
    
    float local_rss = 0.0f;
    if (idx < n) {
        local_rss = residuals[idx] * residuals[idx];
    }
    
    shared_rss[threadIdx.x] = local_rss;
    __syncthreads();
    
    // Redução paralela
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_rss[threadIdx.x] += shared_rss[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(rss, shared_rss[0]);
    }
}

int main() {
    // Parâmetros do exemplo
    const int n = 1000;  // Número de amostras
    const int p = 4;     // Número de coeficientes (incluindo intercepto)
    
    // Alocação de memória no host
    float* h_X = (float*)malloc(n * (p-1) * sizeof(float));
    float* h_beta = (float*)malloc(p * sizeof(float));
    float* h_y_true = (float*)malloc(n * sizeof(float));
    float* h_rss = (float*)malloc(sizeof(float));
    
    // Inicialização dos dados de exemplo
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p-1; j++) {
            h_X[i * (p-1) + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
    
    // Inicialização dos coeficientes β
    for (int j = 0; j < p; j++) {
        h_beta[j] = j + 1;  // Exemplo: 1, 2, 3, 4
    }
    
    // Gerar valores y verdadeiros com ruído
    for (int i = 0; i < n; i++) {
        h_y_true[i] = h_beta[0];  // Intercepto
        for (int j = 1; j < p; j++) {
            h_y_true[i] += h_X[i * (p-1) + (j-1)] * h_beta[j];
        }
        h_y_true[i] += ((float)rand() / RAND_MAX) * 0.1f;  // Adiciona ruído
    }
    
    // Alocação de memória no device
    float *d_X, *d_beta, *d_y_true, *d_Xbeta, *d_residuals, *d_rss;
    cudaMalloc((void**)&d_X, n * (p-1) * sizeof(float));
    cudaMalloc((void**)&d_beta, p * sizeof(float));
    cudaMalloc((void**)&d_y_true, n * sizeof(float));
    cudaMalloc((void**)&d_Xbeta, n * sizeof(float));
    cudaMalloc((void**)&d_residuals, n * sizeof(float));
    cudaMalloc((void**)&d_rss, sizeof(float));
    
    // Cópia dos dados para o device
    cudaMemcpy(d_X, h_X, n * (p-1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_true, h_y_true, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configuração das dimensões do grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Inicializar RSS com zero
    float zero = 0.0f;
    cudaMemcpy(d_rss, &zero, sizeof(float), cudaMemcpyHostToDevice);
    
    // Execução dos kernels
    computeXBeta<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_beta, d_Xbeta, n, p);
    computeResiduals<<<blocksPerGrid, threadsPerBlock>>>(d_y_true, d_Xbeta, d_residuals, n);
    computeRSS<<<blocksPerGrid, threadsPerBlock>>>(d_residuals, d_rss, n);
    
    // Sincronização e cópia dos resultados
    float h_rss_result;
    cudaMemcpy(&h_rss_result, d_rss, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cópia dos resultados intermediários para visualização
    float* h_Xbeta = (float*)malloc(n * sizeof(float));
    float* h_residuals = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_Xbeta, d_Xbeta, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_residuals, d_residuals, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Impressão dos resultados
    printf("\nPrimeiras 5 predições e resíduos:\n");
    for (int i = 0; i < 5; i++) {
        printf("Amostra %d:\n", i);
        printf("  Predição (Xβ): %f\n", h_Xbeta[i]);
        printf("  Valor real (y): %f\n", h_y_true[i]);
        printf("  Resíduo: %f\n", h_residuals[i]);
    }
    
    printf("\nSoma dos Quadrados dos Resíduos (RSS): %f\n", h_rss_result);
    
    // Limpeza da memória
    // Device
    cudaFree(d_X);
    cudaFree(d_beta);
    cudaFree(d_y_true);
    cudaFree(d_Xbeta);
    cudaFree(d_residuals);
    cudaFree(d_rss);
    
    // Host
    free(h_X);
    free(h_beta);
    free(h_y_true);
    free(h_rss);
    free(h_Xbeta);
    free(h_residuals);
    
    return 0;
} 