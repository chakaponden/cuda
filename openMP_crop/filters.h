#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define KERNEL_SIZE 3

pixel_channel kernel[KERNEL_SIZE][KERNEL_SIZE] =		{ -1, -1, -1,
								  -1,  9, -1, 
								  -1, -1, -1 };

								  
								  
double cpu_filter(wind_image **src_img, wind_image** result_img)
{
	clock_t begin, end;
	begin = clock();

	//omp_set_dynamic(0);    
	//omp_set_num_threads(4);
	int rSum, gSum, bSum;
	int i, j, x, y;
	resolution width = (*src_img)->width;
	resolution height = (*src_img)->height;
	(*result_img)->height = height;
	(*result_img)->width = width;
	#pragma omp parallel for shared(src_img, result_img, width, height) private(i, j, x, y, rSum, gSum, bSum)	
	for(y = 1; y < height - 1; y++)
		for(x = 1; x < width - 1; x++)
		{
			rSum = 0, gSum = 0, bSum = 0;

			for (i = 0; i < KERNEL_SIZE; i++)
				for (j = 0; j < KERNEL_SIZE; j++)
				{   
					rSum += ((*src_img)->arrayR[(y + j - 1 ) * width + x + i - 1]) * kernel[j][i];
					gSum += ((*src_img)->arrayG[(y + j - 1 ) * width + x + i - 1]) * kernel[j][i];
					bSum += ((*src_img)->arrayB[(y + j - 1 ) * width + x + i - 1]) * kernel[j][i];
				}

			if (rSum < 0) 
				rSum = 0;
			if (rSum > 255) 
				rSum = 255;
			if (gSum < 0) 
				gSum = 0;
			if (gSum > 255) 
				gSum = 255;
			if (bSum < 0) 
				bSum = 0;
			if (bSum > 255) 
				bSum = 255;

			(*result_img)->arrayR[y * width + x] = rSum;
			(*result_img)->arrayG[y * width + x] = gSum;
			(*result_img)->arrayB[y * width + x] = bSum;				
		}
		end = clock();
		return ((double)(end - begin) / (CLOCKS_PER_SEC / 1000));
}

/*
double cpu_filter(wind_image **src_img, wind_image **result_img)
{
  clock_t begin, end;
  begin = clock();
  resolution width = (*src_img)->width;
  resolution height = (*src_img)->height;
  resolution x, y, i, j, pixelPosX, pixelPosY;
  pixel_channel r, g, b, rSum, gSum, bSum;
  free_image(result_img);
  (*result_img) = (wind_image*)malloc(sizeof(wind_image));
  (*result_img)->height = height;
  (*result_img)->width = width;
  (*result_img)->arrayR = (pixel_channel*)malloc(width * height * sizeof(pixel_channel));
  (*result_img)->arrayG = (pixel_channel*)malloc(width * height * sizeof(pixel_channel));
  (*result_img)->arrayB = (pixel_channel*)malloc(width * height * sizeof(pixel_channel));
  for(x = 0; x < width; x++)
  for(y = 0; y < height; y++)
  {
    rSum = 0, gSum = 0, bSum = 0;
    for(i = 0; i < KERNEL_SIZE; i++)
      for(j = 0; j < KERNEL_SIZE; j++)
      {
	pixelPosX = x + i - 1;
	pixelPosY = y + j - 1;
	if ((pixelPosX < 0) || (pixelPosX >= width) | (pixelPosY < 0) || (pixelPosY >= height))
	  continue;
	r = (*src_img)->arrayR[(width * pixelPosY + pixelPosX)];
	g = (*src_img)->arrayG[(width * pixelPosY + pixelPosX)];
	b = (*src_img)->arrayB[(width * pixelPosY + pixelPosX)];
	rSum += r * kernel[i][j];
	gSum += g * kernel[i][j];
	bSum += b * kernel[i][j];
      }
      if (rSum < 0) rSum = 0;
      if (rSum > 255) rSum = 255;
      if (gSum < 0) gSum = 0;
      if (gSum > 255) gSum = 255;
      if (bSum < 0) bSum = 0;
      if (bSum > 255) bSum = 255;
      (*result_img)->arrayR[(width * y + x)] = (pixel_channel)rSum;
      (*result_img)->arrayG[(width * y + x)] = (pixel_channel)gSum;
      (*result_img)->arrayB[(width * y + x)] = (pixel_channel)bSum;
  }
  end = clock();
  return ((double)(end - begin) / (CLOCKS_PER_SEC/1000));
}
*/

double async_cuda_filter(wind_image **src_img)
{
  clock_t begin, end;  
  begin = clock();
  pixel_channel* RGB[3];
  RGB[0] = (*src_img)->arrayR;
  RGB[1] = (*src_img)->arrayG;
  RGB[2] = (*src_img)->arrayB;
  asyncConvolution(RGB, (*src_img)->width, (*src_img)->height);
  (*src_img)->arrayR = RGB[0];
  (*src_img)->arrayG = RGB[1];
  (*src_img)->arrayB = RGB[2];
  end = clock();
  return ((double)(end - begin) / (CLOCKS_PER_SEC/1000));  
}

unsigned char *result_bmp;
wind_image *cpu_image, *cpu_image_result, *gpu_image;
pthread_t cpu_thread, gpu_thread;
double cpu_time, gpu_time, gpu_charge_time;
volatile int running_threads = 0;
pthread_mutex_t running_mutex = PTHREAD_MUTEX_INITIALIZER;

void* cpu_filter_thread(void *arg)
{
  //printf("cpu_filter start\n");
  cpu_time = cpu_filter(&cpu_image, &cpu_image_result);
  pthread_mutex_lock(&running_mutex);
  running_threads--;
  pthread_mutex_unlock(&running_mutex);
  //printf("cpu_filter end\n");
  return NULL;
}

void* gpu_filter_thread(void *arg)
{
  //printf("gpu_filter start\n");
  gpu_time = async_cuda_filter(&gpu_image);
  pthread_mutex_lock(&running_mutex);
  running_threads--;
  pthread_mutex_unlock(&running_mutex);
  //printf("gpu_filter end\n");
  return NULL;
}