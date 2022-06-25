// Toy program to exercise gpu offloading.
// https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/
//
// CUDA functions:
//  cudaMalloc(void **devPtr, size_t count);
//  cudaFree(void *devPtr);
//  cudaMemcpy(void *dst, void *src, size_t count, cudaMemcpyKind kind)
//  cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    float *a = (float*)malloc(sizeof(*a) * N);
    float *b = (float*)malloc(sizeof(*b) * N);
    float *out = (float*)malloc(sizeof(*out) * N);

    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    memset(out, 0, sizeof(*out) * N);

    float *gpu_a;
    float *gpu_b;
    float *gpu_out;
    cudaMalloc((void**)&gpu_a, sizeof(*gpu_a) * N);
    cudaMalloc((void**)&gpu_b, sizeof(*gpu_b) * N);
    cudaMalloc((void**)&gpu_out, sizeof(*gpu_out) * N);

    cudaMemcpy(gpu_a, a, sizeof(*a) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, sizeof(*b) * N, cudaMemcpyHostToDevice);

    vector_add<<<1,1>>>(gpu_out, gpu_a, gpu_b, N);

    cudaMemcpy(out, gpu_out, sizeof(*out) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        if (fabs(out[i] - a[i] - b[i]) >= MAX_ERR) {
            fprintf(
                stderr,
                "out[%d]=%f - a[%d]=%f - b[%d]=%f > %f\n",
                i,
                out[i],
                i,
                a[i],
                i,
                b[i],
                MAX_ERR);
            assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
        }
    }
    fprintf(stderr, "out[0] = %f\n", out[0]);
    fprintf(stderr, "PASSED\n");

    free(a);
    free(b);
    free(out);

    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_out);

    return 0;
}
