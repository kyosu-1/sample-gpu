// vector_sqrt_compare.cu
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

__global__ void vectorSqrt(const float *A, float *B, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        B[i] = sqrt(A[i]);
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

// CPUでの平方根計算
void vectorSqrtCPU(const float *A, float *B, int N) {
    for (int i = 0; i < N; i++) {
        B[i] = sqrt(A[i]);
    }
}

int main() {
    int N = 1<<25; // 33M elements
    size_t size = N * sizeof(float);

    // Allocate memory on the host
    float *h_A = (float*)malloc(size);
    float *h_B_CPU = (float*)malloc(size);
    float *h_B_GPU = (float*)malloc(size);

    // Initialize vectors on the host
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
    }

    // Allocate memory on the device
    float *d_A, *d_B;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_B, size));

    // Copy vector from host memory to device memory
    CUDA_CHECK_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // CPUでの計算
    auto start = std::chrono::high_resolution_clock::now();
    vectorSqrtCPU(h_A, h_B_CPU, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end - start;
    std::cout << "CPU time: " << cpu_duration.count() << " seconds" << std::endl;

    // GPUでの計算
    start = std::chrono::high_resolution_clock::now();
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorSqrt<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    CUDA_CHECK_ERROR(cudaGetLastError()); // カーネル呼び出し後のエラーチェック
    CUDA_CHECK_ERROR(cudaDeviceSynchronize()); // カーネル呼び出し後の同期
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = end - start;
    std::cout << "GPU time: " << gpu_duration.count() << " seconds" << std::endl;

    // Copy the result from device memory to host memory
    CUDA_CHECK_ERROR(cudaMemcpy(h_B_GPU, d_B, size, cudaMemcpyDeviceToHost));

    // Verify the result and print some of the computation
    bool verification_passed = true;
    for (int i = 0; i < N; i++) {
        if (i < 10 || i == N-1) { // 最初の10要素と最後の1要素を表示
            std::cout << "Element " << i << ": sqrt(" << h_A[i] << ") = " << h_B_GPU[i] << " (GPU), " << h_B_CPU[i] << " (CPU)" << std::endl;
        }
        if (fabs(h_B_GPU[i] - h_B_CPU[i]) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << std::endl;
            verification_passed = false;
        }
    }

    if (verification_passed) {
        std::cout << "Test PASSED" << std::endl;
    }

    // Free device memory
    CUDA_CHECK_ERROR(cudaFree(d_A));
    CUDA_CHECK_ERROR(cudaFree(d_B));

    // Free host memory
    free(h_A);
    free(h_B_CPU);
    free(h_B_GPU);

    return 0;
}

