#include <cuda.h>

#define KERNEL_SIZE 3
#define BLOCK_SIZE  512

typedef signed int pixel_channel;
typedef unsigned long resolution;

__constant__ pixel_channel kernel_cuda[KERNEL_SIZE * KERNEL_SIZE];
pixel_channel kernel_host[KERNEL_SIZE * KERNEL_SIZE] = 		{ -1, -1, -1,
								  -1,  9, -1, 
								  -1, -1, -1 };


__global__ void Pixel_Shared_Convolution(pixel_channel *channel_cuda, pixel_channel *rezult_cuda, resolution width, resolution lineQuantity)
{
    __shared__ pixel_channel sharedMemory [3][BLOCK_SIZE + 2];

    for(long line = 1; line < lineQuantity; line++)
    {
        long temp = blockIdx.x * BLOCK_SIZE + threadIdx.x;

        sharedMemory [0][threadIdx.x + 1] = channel_cuda[temp + width * (line - 1)];
        sharedMemory [1][threadIdx.x + 1] = channel_cuda[temp + width * line];
        sharedMemory [2][threadIdx.x + 1] = channel_cuda[temp + width * (line + 1)];

        if(threadIdx.x == 0)
        {
			if(blockIdx.x != 0)
				temp--;
            sharedMemory [0][0] = channel_cuda[temp + width * (line-1)];
            sharedMemory [1][0] = channel_cuda[temp + width * line];
            sharedMemory [2][0] = channel_cuda[temp + width * (line+1)];
        }

        if(threadIdx.x == (BLOCK_SIZE - 1))
        {
            temp++;
            sharedMemory [0][BLOCK_SIZE + 1] = channel_cuda[temp + width * (line - 1)];
            sharedMemory [1][BLOCK_SIZE + 1] = channel_cuda[temp + width * line];
            sharedMemory [2][BLOCK_SIZE + 1] = channel_cuda[temp + width * (line + 1)];
        }
        __syncthreads();

        long Sum = 0;

        for (int i = 0; i < KERNEL_SIZE; i++)
            for (int j = 0; j < KERNEL_SIZE; j++)
                Sum += sharedMemory[j][threadIdx.x + i] * kernel_cuda[i * 3 + j];

        if (Sum < 0)
            Sum = 0;
        if (Sum > 255)
            Sum = 255;

		__syncthreads();
		
		if((blockIdx.x * BLOCK_SIZE + threadIdx.x) > width)
			continue;
        
		rezult_cuda[blockIdx.x * BLOCK_SIZE + threadIdx.x + width * line] = Sum;
    }
	__syncthreads();

    return;
}

extern "C" __host__ pixel_channel** asyncConvolution(pixel_channel **image, resolution width, resolution height)
{
    pixel_channel **channel_cuda;
    channel_cuda = (pixel_channel**)malloc(3*sizeof(pixel_channel*));

    pixel_channel **rezult_cuda;
    rezult_cuda = (pixel_channel**)malloc(3*sizeof(pixel_channel*));

    resolution size = width * height;

    cudaHostRegister(image[0], (size + BLOCK_SIZE) * sizeof(pixel_channel), cudaHostRegisterMapped);
    cudaHostRegister(image[1], (size + BLOCK_SIZE) * sizeof(pixel_channel), cudaHostRegisterMapped);
    cudaHostRegister(image[2], (size + BLOCK_SIZE) * sizeof(pixel_channel), cudaHostRegisterMapped);

    cudaMalloc((void **)& rezult_cuda[0], (size + BLOCK_SIZE) * sizeof(pixel_channel));
    cudaMalloc((void **)& rezult_cuda[1], (size + BLOCK_SIZE) * sizeof(pixel_channel));
    cudaMalloc((void **)& rezult_cuda[2], (size + BLOCK_SIZE) * sizeof(pixel_channel));

    cudaMalloc((void **)& channel_cuda[0], (size + BLOCK_SIZE) * sizeof(pixel_channel));;
    cudaMalloc((void **)& channel_cuda[1], (size + BLOCK_SIZE) * sizeof(pixel_channel));
    cudaMalloc((void **)& channel_cuda[2], (size + BLOCK_SIZE) * sizeof(pixel_channel));

    cudaMemcpyToSymbol(kernel_cuda, kernel_host, 9 * sizeof(pixel_channel), 0, cudaMemcpyHostToDevice);

    resolution block_count = 0;
    if(((width - 2)%BLOCK_SIZE) == 0)
        block_count = (width - 2)/BLOCK_SIZE;
    else
        block_count = (width - 2)/BLOCK_SIZE + 1;

    dim3 gridSize = dim3(block_count, 1, 1);
    dim3 blockSize = dim3(BLOCK_SIZE, 1, 1);

    cudaStream_t stream[3];

    for(int i = 0; i < 3; i++)
    {
	cudaStreamCreate(&stream[i]);
        cudaMemcpyAsync(channel_cuda[i], image[i], size*sizeof(pixel_channel), cudaMemcpyHostToDevice, stream[i]);
        Pixel_Shared_Convolution<<<gridSize, blockSize, 0, stream[i]>>>(channel_cuda[i], rezult_cuda[i], width, height);
        cudaMemcpyAsync(image[i], rezult_cuda[i], size*sizeof(pixel_channel), cudaMemcpyDeviceToHost,stream[i]);
	cudaStreamDestroy(stream[i]);
    }

    for(int i=0;i<3;i++)
    {
        cudaFree(rezult_cuda[i]);
        cudaFree(channel_cuda[i]);
    }

    cudaDeviceReset();

    return image;
}
