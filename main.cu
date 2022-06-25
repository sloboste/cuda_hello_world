// Toy program to exercise gpu offloading.
// https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/
// https://docs.nvidia.com/cuda/cuda-runtime-api/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define TOTAL_SIZE (1 << 30)
#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS ((TOTAL_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)

#define MAX_ERR 1e-6

#define STRINGIFY(x) #x

#define cuda_try(call) \
    do { \
        cudaError_t ret = call; \
        if (ret != 0) { \
            on_cuda_error_(ret, STRINGIFY(call)); \
        } \
    } while (false)

void on_cuda_error_(cudaError_t error, const char *call) {
    const char *name = cudaGetErrorName(error);
    const char *description = cudaGetErrorString(error);
    fprintf(stderr, "%s failed with %s:\n\t%s\n", call, name, description);
    exit(1);
}

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;;
    }
    out[tid] = a[tid] + b[tid];
}

void do_gpu_stuff() {
    size_t mem_size = sizeof(float) * TOTAL_SIZE;
    float *a, *b, *out;
    cuda_try(cudaMallocManaged(&a, mem_size));
    cuda_try(cudaMallocManaged(&b, mem_size));
    cuda_try(cudaMallocManaged(&out, mem_size));

    for (int i = 0; i < TOTAL_SIZE; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    memset(out, 0, mem_size);

    vector_add<<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(out, a, b, TOTAL_SIZE);
    cuda_try(cudaDeviceSynchronize());

    for (int i = 0; i < TOTAL_SIZE; ++i) {
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
    fprintf(stderr, "PASSED\n");
}

void print_gpu_properties(cudaDeviceProp *p) {
    printf(
        "name=%s\n" // ASCII string identifying device
        // FIXME "uuid=%16s\n" // 16-byte unique identifier
        // "luid=%8s\n" // 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms
        // "luidDeviceNodeMask=%lu\n" // LUID device node mask. Value is undefined on TCC and non-Windows platforms
        "totalGlobalMem=%zu\n" // Global memory available on device in bytes
        "sharedMemPerBlock=%zu\n" // Shared memory available per block in bytes
        "regsPerBlock=%d\n" // 32-bit registers available per block
        "warpSize=%d\n" // Warp size in threads
        "memPitch=%zu\n" // Maximum pitch in bytes allowed by memory copies
        "maxThreadsPerBlock=%d\n" // Maximum number of threads per block
        "maxThreadsDim[3]=[%d, %d, %d]\n" // Maximum size of each dimension of a block
        "maxGridSize[3]=[%d, %d, %d]\n" // Maximum size of each dimension of a grid
        "clockRate=%d\n" // Clock frequency in kilohertz
        "totalConstMem=%zu\n" // Constant memory available on device in bytes
        "major=%d\n" // Major compute capability
        "minor=%d\n" // Minor compute capability
        "textureAlignment=%zu\n" // Alignment requirement for textures
        "texturePitchAlignment=%zu\n" // Pitch alignment requirement for texture references bound to pitched memory
        "deviceOverlap=%d\n" // Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount.
        "multiProcessorCount=%d\n" // Number of multiprocessors on device
        "kernelExecTimeoutEnabled=%d\n" // Specified whether there is a run time limit on kernels
        "integrated=%d\n" // Device is integrated as opposed to discrete
        "canMapHostMemory=%d\n" // Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
        "computeMode=%d\n" // Compute mode (See ::cudaComputeMode)
        "maxTexture1D=%d\n" // Maximum 1D texture size
        "maxTexture1DMipmap=%d\n" // Maximum 1D mipmapped texture size
        "maxTexture1DLinear=%d\n" // Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead.
        "maxTexture2D[2]=[%d, %d]\n" // Maximum 2D texture dimensions
        "maxTexture2DMipmap[2]=[%d, %d]\n" // Maximum 2D mipmapped texture dimensions
        "maxTexture2DLinear[3]=[%d, %d, %d]\n" // Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory
        "maxTexture2DGather[2]=[%d, %d]\n" // Maximum 2D texture dimensions if texture gather operations have to be performed
        "maxTexture3D[3]=[%d, %d, %d]\n" // Maximum 3D texture dimensions
        "maxTexture3DAlt[3]=[%d, %d, %d]\n" // Maximum alternate 3D texture dimensions
        "maxTextureCubemap=%d\n" // Maximum Cubemap texture dimensions
        "maxTexture1DLayered[2]=[%d, %d]\n" // Maximum 1D layered texture dimensions
        "maxTexture2DLayered[3]=[%d, %d, %d]\n" // Maximum 2D layered texture dimensions
        "maxTextureCubemapLayered[2]=[%d, %d]\n" // Maximum Cubemap layered texture dimensions
        "maxSurface1D=%d\n" // Maximum 1D surface size
        "maxSurface2D[2]=[%d, %d]\n" // Maximum 2D surface dimensions
        "maxSurface3D[3]=[%d, %d, %d]\n" // Maximum 3D surface dimensions
        "maxSurface1DLayered[2]=[%d, %d]\n" // Maximum 1D layered surface dimensions
        "maxSurface2DLayered[3]=[%d, %d, %d]\n" // Maximum 2D layered surface dimensions
        "maxSurfaceCubemap=%d\n" // Maximum Cubemap surface dimensions
        "maxSurfaceCubemapLayered[2]=[%d, %d]\n" // Maximum Cubemap layered surface dimensions
        "surfaceAlignment=%zu\n" // Alignment requirements for surfaces
        "concurrentKernels=%d\n" // Device can possibly execute multiple kernels concurrently
        "ECCEnabled=%d\n" // Device has ECC support enabled
        "pciBusID=%d\n" // PCI bus ID of the device
        "pciDeviceID=%d\n" // PCI device ID of the device
        "pciDomainID=%d\n" // PCI domain ID of the device
        "tccDriver=%d\n" // 1 if device is a Tesla device using TCC driver, 0 otherwise
        "asyncEngineCount=%d\n" // Number of asynchronous engines
        "unifiedAddressing=%d\n" // Device shares a unified address space with the host
        "memoryClockRate=%d\n" // Peak memory clock frequency in kilohertz
        "memoryBusWidth=%d\n" // Global memory bus width in bits
        "l2CacheSize=%d\n" // Size of L2 cache in bytes
        "persistingL2CacheMaxSize=%d\n" // Device's maximum l2 persisting lines capacity setting in bytes
        "maxThreadsPerMultiProcessor=%d\n" // Maximum resident threads per multiprocessor
        "streamPrioritiesSupported=%d\n" // Device supports stream priorities
        "globalL1CacheSupported=%d\n" // Device supports caching globals in L1
        "localL1CacheSupported=%d\n" // Device supports caching locals in L1
        "sharedMemPerMultiprocessor=%zu\n" // Shared memory available per multiprocessor in bytes
        "regsPerMultiprocessor=%d\n" // 32-bit registers available per multiprocessor
        "managedMemory=%d\n" // Device supports allocating managed memory on this system
        "isMultiGpuBoard=%d\n" // Device is on a multi-GPU board
        "multiGpuBoardGroupID=%d\n" // Unique identifier for a group of devices on the same multi-GPU board
        "hostNativeAtomicSupported=%d\n" // Link between the device and the host supports native atomic operations
        "singleToDoublePrecisionPerfRatio=%d\n" // Ratio of single precision performance (in floating-point operations per second) to double precision performance
        "pageableMemoryAccess=%d\n" // Device supports coherently accessing pageable memory without calling cudaHostRegister on it
        "concurrentManagedAccess=%d\n" // Device can coherently access managed memory concurrently with the CPU
        "computePreemptionSupported=%d\n" // Device supports Compute Preemption
        "canUseHostPointerForRegisteredMem=%d\n" // Device can access host registered memory at the same virtual address as the CPU
        "cooperativeLaunch=%d\n" // Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel
        "cooperativeMultiDeviceLaunch=%d\n" // Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated.
        "sharedMemPerBlockOptin=%zu\n" // Per device maximum shared memory per block usable by special opt in
        "pageableMemoryAccessUsesHostPageTables=%d\n" // Device accesses pageable memory via the host's page tables
        "directManagedMemAccessFromHost=%d\n" // Host can directly access managed memory on the device without migration.
        "maxBlocksPerMultiProcessor=%d\n" // Maximum number of resident blocks per multiprocessor
        "accessPolicyMaxWindowSize=%d\n" // The maximum value of ::cudaAccessPolicyWindow::num_bytes.
        "reservedSharedMemPerBlock=%zu\n" // Shared memory reserved by CUDA driver per block in bytes
        "\n",
        p->name,
        // FIXME p->uuid.bytes,
        // p->luid,
        // p->luidDeviceNodeMask,
        p->totalGlobalMem,
        p->sharedMemPerBlock,
        p->regsPerBlock,
        p->warpSize,
        p->memPitch,
        p->maxThreadsPerBlock,
        p->maxThreadsDim[0], p->maxThreadsDim[1], p->maxThreadsDim[2],
        p->maxGridSize[0], p->maxGridSize[2], p->maxGridSize[2],
        p->clockRate,
        p->totalConstMem,
        p->major,
        p->minor,
        p->textureAlignment,
        p->texturePitchAlignment,
        p->deviceOverlap,
        p->multiProcessorCount,
        p->kernelExecTimeoutEnabled,
        p->integrated,
        p->canMapHostMemory,
        p->computeMode,
        p->maxTexture1D,
        p->maxTexture1DMipmap,
        p->maxTexture1DLinear,
        p->maxTexture2D[0], p->maxTexture2D[1],
        p->maxTexture2DMipmap[0], p->maxTexture2DMipmap[1],
        p->maxTexture2DLinear[0], p->maxTexture2DLinear[1], p->maxTexture2DLinear[2],
        p->maxTexture2DGather[0], p->maxTexture2DGather[1],
        p->maxTexture3D[0], p->maxTexture3D[1], p->maxTexture3D[2],
        p->maxTexture3DAlt[0], p->maxTexture3DAlt[1], p->maxTexture3DAlt[2],
        p->maxTextureCubemap,
        p->maxTexture1DLayered[0], p->maxTexture1DLayered[1],
        p->maxTexture2DLayered[0], p->maxTexture2DLayered[1], p->maxTexture2DLayered[2],
        p->maxTextureCubemapLayered[0], p->maxTextureCubemapLayered[1],
        p->maxSurface1D,
        p->maxSurface2D[0], p->maxSurface2D[1],
        p->maxSurface3D[0], p->maxSurface3D[1], p->maxSurface3D[2],
        p->maxSurface1DLayered[0], p->maxSurface1DLayered[1],
        p->maxSurface2DLayered[0], p->maxSurface2DLayered[1], p->maxSurface2DLayered[2],
        p->maxSurfaceCubemap,
        p->maxSurfaceCubemapLayered[0], p->maxSurfaceCubemapLayered[1],
        p->surfaceAlignment,
        p->concurrentKernels,
        p->ECCEnabled,
        p->pciBusID,
        p->pciDeviceID,
        p->pciDomainID,
        p->tccDriver,
        p->asyncEngineCount,
        p->unifiedAddressing,
        p->memoryClockRate,
        p->memoryBusWidth,
        p->l2CacheSize,
        p->persistingL2CacheMaxSize,
        p->maxThreadsPerMultiProcessor,
        p->streamPrioritiesSupported,
        p->globalL1CacheSupported,
        p->localL1CacheSupported,
        p->sharedMemPerMultiprocessor,
        p->regsPerMultiprocessor,
        p->managedMemory,
        p->isMultiGpuBoard,
        p->multiGpuBoardGroupID,
        p->hostNativeAtomicSupported,
        p->singleToDoublePrecisionPerfRatio,
        p->pageableMemoryAccess,
        p->concurrentManagedAccess,
        p->computePreemptionSupported,
        p->canUseHostPointerForRegisteredMem,
        p->cooperativeLaunch,
        p->cooperativeMultiDeviceLaunch,
        p->sharedMemPerBlockOptin,
        p->pageableMemoryAccessUsesHostPageTables,
        p->directManagedMemAccessFromHost,
        p->maxBlocksPerMultiProcessor,
        p->accessPolicyMaxWindowSize,
        p->reservedSharedMemPerBlock);
}

int main(int argc, char **argv) {
    bool bad_args = true;
    bool verbose = false;
    if (argc == 1) {
        bad_args = false;
    } else if (argc == 2 && strcmp(argv[1], "-v") == 0) {
        bad_args = false;
        verbose = true;
    }
    if (bad_args) {
        fprintf(stderr, "Unrecognized argument(s)");
        return 1;
    }

    if (verbose) {
        int device_count;
        cuda_try(cudaGetDeviceCount(&device_count));
        printf("Detected %d CUDA-capable devices\n\n", device_count);

        cudaDeviceProp properties;
        cuda_try(cudaGetDeviceProperties(&properties, 0));
        print_gpu_properties(&properties);
    }

    do_gpu_stuff();
    return 0;
}
