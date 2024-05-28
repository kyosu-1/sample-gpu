// vector_add.cu
#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// CUDAエラーチェックマクロ
#define CUDA_CHECK_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    int N = 1<<20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate memory on the host
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize vectors on the host
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_C, size));

    // Copy vectors from host memory to device memory
    CUDA_CHECK_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK_ERROR(cudaGetLastError()); // カーネル呼び出し後のエラーチェック
    CUDA_CHECK_ERROR(cudaDeviceSynchronize()); // カーネル呼び出し後の同期

    // Copy the result from device memory to host memory
    CUDA_CHECK_ERROR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Test PASSED" << std::endl;

    // Free device memory
    CUDA_CHECK_ERROR(cudaFree(d_A));
    CUDA_CHECK_ERROR(cudaFree(d_B));
    CUDA_CHECK_ERROR(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

