#include <cuda.h>

#define KERNEL_SIZE 3
#define BLOCK_SIZE  512

typedef signed int pixel_channel;
typedef unsigned long resolution;

__constant__ pixel_channel kernel_cuda[KERNEL_SIZE * KERNEL_SIZE];
pixel_channel kernel_host[KERNEL_SIZE * KERNEL_SIZE];

__global__ void Pixel_Shared_Convolution(pixel_channel *channel_cuda, pixel_channel *rezult_cuda, resolution width, resolution lineQuantity)
{
    __shared__ pixel_channel sharedMemory [3][BLOCK_SIZE + 2];

    for(resolution line = 1; line < lineQuantity; line++)
    {
        resolution temp = blockIdx.x * BLOCK_SIZE + threadIdx.x + 1;

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

extern "C" __host__ void Shared_Memory_Convolution(pixel_channel **channel, resolution width, resolution height, pixel_channel kernel[3][3])
{
    pixel_channel *channel_cuda, *rezult_cuda;
    resolution size = width * height;

    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            kernel_host[i * 3 + j] = kernel[i][j];

    resolution block_count = 0;
    if(((width - 2)%BLOCK_SIZE) == 0)
        block_count = (width - 2)/BLOCK_SIZE;
    else
        block_count = (width - 2)/BLOCK_SIZE + 1;

    dim3 gridSize = dim3(block_count, 1, 1);
    dim3 blockSize = dim3(BLOCK_SIZE, 1, 1);

    cudaMalloc((void **)& rezult_cuda, (size + 256) * sizeof(pixel_channel));
    cudaMalloc((void **)& channel_cuda, (size + 256) * sizeof(pixel_channel));

    cudaMemcpy(channel_cuda, *channel, size * sizeof(pixel_channel), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel_cuda, kernel_host, 9 * sizeof(pixel_channel), 0, cudaMemcpyHostToDevice);

    Pixel_Shared_Convolution<<<gridSize, blockSize>>>(channel_cuda, rezult_cuda, width, height - 2);

    cudaMemcpy(*channel, rezult_cuda, size * sizeof(pixel_channel), cudaMemcpyDeviceToHost);

    cudaFree(rezult_cuda);
    cudaFree(channel_cuda);

    cudaDeviceReset();

    return;
}

__global__ void new_convolution(int *channel_cuda, int *rezult_cuda, int width, int height, int size, int lineQuantity)
{
    __shared__ int sharedMemory [3][BLOCK_SIZE+2];

    for(int line=1; line<lineQuantity; line++)
    {
        int temp = blockIdx.x * BLOCK_SIZE + threadIdx.x + 1;

        sharedMemory [0][threadIdx.x+1] = channel_cuda[temp + width * (line-1)];
        sharedMemory [1][threadIdx.x+1] = channel_cuda[temp + width * line];
        sharedMemory [2][threadIdx.x+1] = channel_cuda[temp + width * (line+1)];
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
            sharedMemory [0][BLOCK_SIZE] = channel_cuda[temp + width * (line-1)];
            sharedMemory [1][BLOCK_SIZE] = channel_cuda[temp + width * line + 2];
            sharedMemory [2][BLOCK_SIZE] = channel_cuda[temp + width * (line+1)];
        }
        __syncthreads();

        int rSum = 0, kSum = 0, kernelVal, r;

        for (int i = 0; i < KERNEL_SIZE; i++)
        {
            for (int j = 0; j < KERNEL_SIZE; j++)
            {
                r = sharedMemory[j][threadIdx.x+1];

                kernelVal = kernel_cuda[i*3+j];                     //Получаем значение ядра
                rSum += r * kernelVal;
                kSum += kernelVal;
            }
        }

        if (kSum <= 0) kSum = 1;

        //Контролируем переполнения переменных
        rSum /= kSum;
        if (rSum < 0) rSum = 0;
        if (rSum > 255) rSum = 255;


        //Записываем значения в результирующее изображение

        //if((blockIdx.x * BLOCK_SIZE + threadIdx.x) < (width - 1))
        rezult_cuda[blockIdx.x * BLOCK_SIZE + threadIdx.x + width * line + 1] = rSum;



    }
}

extern "C" __host__ void asyncConvolution(int **image, int width, int height)
{
    #define STREAM_QUANTITY 3
    int **channel_cuda; channel_cuda = (int**)malloc(3*sizeof(int*));
    int **rezult_cuda; rezult_cuda = (int**)malloc(3*sizeof(int*));

    int size = width * height;                                      //Размер изображения

    kernel_host[0] = -1;     //Инициализация ядра
    kernel_host[1] = -1;
    kernel_host[2] = -1;
    kernel_host[3] = -1;     // -1 -1 -1
    kernel_host[4] =  9;     // -1 9  -1
    kernel_host[5] = -1;     // -1 -1 -1
    kernel_host[6] = -1;
    kernel_host[7] = -1;
    kernel_host[8] = -1;

    //Преобразовываем память в pinned-память
    cudaHostRegister(image[0], (width * height + 256) * sizeof(int), cudaHostRegisterMapped);
    cudaHostRegister(image[1], (width * height + 256) * sizeof(int), cudaHostRegisterMapped);
    cudaHostRegister(image[2], (width * height + 256) * sizeof(int), cudaHostRegisterMapped);

    cudaMalloc((void **)& rezult_cuda[0],      (width * height + 256) * sizeof(int));
    cudaMalloc((void **)& rezult_cuda[1],    (width * height + 256) * sizeof(int));
    cudaMalloc((void **)& rezult_cuda[2],     (width * height + 256) * sizeof(int));

    cudaMalloc((void **)& channel_cuda[0],     (width * height + 256) * sizeof(int));
    cudaMalloc((void **)& channel_cuda[1],   (width * height + 256) * sizeof(int));
    cudaMalloc((void **)& channel_cuda[2],    (width * height + 256) * sizeof(int));

    //Копируем константную память
    cudaMemcpyToSymbol(kernel_cuda,kernel_host,9*sizeof(int),0,cudaMemcpyHostToDevice);

    dim3 gridSize = dim3((width - 2)/BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize = dim3(BLOCK_SIZE, 1, 1);

    cudaStream_t    stream[STREAM_QUANTITY];
    for(int i=0;i<STREAM_QUANTITY;i++)
    {
        cudaStreamCreate(&stream[i]);
    }
    for(int i=0;i<STREAM_QUANTITY;i++)
    {
        cudaMemcpyAsync(channel_cuda[i],    image[i]    ,width*height*sizeof(int),cudaMemcpyHostToDevice,stream[i]);

    }

    for(int i=0;i<STREAM_QUANTITY;i++)
    {

        new_convolution<<<gridSize,blockSize,0,stream[i]>>>(channel_cuda[i],   rezult_cuda[i],    width,height,size,height-2);
    }

    for(int i=0;i<STREAM_QUANTITY;i++)
        cudaMemcpyAsync(image[i],    rezult_cuda[i],    width*height*sizeof(int),cudaMemcpyDeviceToHost,stream[i]);

    for(int i=0;i<STREAM_QUANTITY;i++)
        cudaStreamDestroy(stream[i]);

    //Очистка памяти
    for(int i=0;i<STREAM_QUANTITY;i++)
    {
        cudaFree(rezult_cuda[i]);
        cudaFree(channel_cuda[i]);
    }

    cudaDeviceReset();

    return;
}
