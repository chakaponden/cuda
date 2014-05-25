#include <cuda.h>
#include <cuda_runtime.h>

#define KERNEL_SIZE 3
#define BLOCK_SIZE  512

typedef signed int pixel_channel;

__constant__ pixel_channel kernel_cuda[KERNEL_SIZE * KERNEL_SIZE];
pixel_channel kernel_host[KERNEL_SIZE * KERNEL_SIZE];

__global__ void Pixel_Shared_Convolution(pixel_channel *channel_cuda, pixel_channel *rezult_cuda, unsigned long width, unsigned long lineQuantity)
{
  
    __shared__ pixel_channel sharedMemory [3][BLOCK_SIZE + 2];

    for(long line = 1; line < lineQuantity; line++)
    {
        long temp = blockIdx.x * BLOCK_SIZE + threadIdx.x + 1;

        sharedMemory [0][threadIdx.x+1] = channel_cuda[temp + width * (line - 1)];
        sharedMemory [1][threadIdx.x+1] = channel_cuda[temp + width * line];
        sharedMemory [2][threadIdx.x+1] = channel_cuda[temp + width * (line + 1)];

        if(threadIdx.x == 0)
        {
            temp--;
            sharedMemory [0][0] = channel_cuda[temp + width * (line-1)];
            sharedMemory [1][0] = channel_cuda[temp + width * line];
            sharedMemory [2][0] = channel_cuda[temp + width * (line+1)];
        }

        if(threadIdx.x == (BLOCK_SIZE-1))
        {
            temp += 2;
            sharedMemory [0][BLOCK_SIZE] = channel_cuda[temp + width * (line - 1)];
            sharedMemory [1][BLOCK_SIZE] = channel_cuda[temp + width * line + 2];
            sharedMemory [2][BLOCK_SIZE] = channel_cuda[temp + width * (line + 1)];
        }
        __syncthreads();

        pixel_channel Sum = 0;

        for (int i = 0; i < KERNEL_SIZE; i++)
            for (int j = 0; j < KERNEL_SIZE; j++)
                Sum += sharedMemory[j][threadIdx.x+1] * kernel_cuda[i * 3 + j];

        if (Sum < 0)
            Sum = 0;
        if (Sum > 255)
            Sum = 255;

        rezult_cuda[blockIdx.x * BLOCK_SIZE + threadIdx.x + width * line + 1] = Sum;

    }

    return;
    
}

extern "C" pixel_channel* Shared_Memory_Convolution(pixel_channel *channel, unsigned long width, unsigned long height, pixel_channel kernel[3][3], float *time)
{
  
    pixel_channel *channel_cuda, *rezult_cuda;
    pixel_channel size = width * height;

    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            kernel_host[i * 3 + j] = kernel[i][j];

    long block_count = 0;
    if(((width - 2)%BLOCK_SIZE) == 0)
        block_count = (width - 2)/BLOCK_SIZE;
    else
        block_count = (width - 2)/BLOCK_SIZE + 1;

    dim3 gridSize = dim3(block_count, 1, 1);
    dim3 blockSize = dim3(BLOCK_SIZE, 1, 1);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaMalloc((void **)& rezult_cuda, (size + 256) * sizeof(pixel_channel));
    cudaMalloc((void **)& channel_cuda, (size + 256) * sizeof(pixel_channel));

    cudaMemcpy(channel_cuda, channel, size * sizeof(pixel_channel), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel_cuda, kernel_host, 9 * sizeof(pixel_channel), 0, cudaMemcpyHostToDevice);

    Pixel_Shared_Convolution<<<gridSize, blockSize>>>(channel_cuda, rezult_cuda, width, (height - 2));

    cudaMemcpy(channel, rezult_cuda, size * sizeof(pixel_channel), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(time, start, stop);

    cudaFree(rezult_cuda);
    cudaFree(channel_cuda);

    cudaDeviceReset();
    

    return channel;
}