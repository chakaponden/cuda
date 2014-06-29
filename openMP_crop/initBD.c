/*
 * [COMPILE]:
 * gcc -c initBD.c -o initBD
 * 
 * [RUN]:
 * ./initBD </full_path/BDfile> <image_height-width> <step>
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "struct_wind_image.h"
#include "filters.h"
#define FILE_NAME_MAX_LEN 256
#define BIN_EXTENSION ".bin"
#define TXT_EXTENSION ".txt"

#define ARG_ERROR_MESS	"[RUN]:\n./initBD </full_path/BDfile> <image_height-width> <step>\n"

double multiplier = 1.0;

double calculate_multiplier(char *test_image_full_path)
{
  gpu_time = -1.0;
  cpu_time = 1.0;
  running_threads = 0;
  pthread_mutex_lock(&running_mutex);
  running_threads++;
  pthread_mutex_unlock(&running_mutex);
  if(pthread_create(&cpu_thread, NULL, cpu_filter_thread, NULL) != 0)
  {
    perror("create cpu_thread");
    pthread_mutex_lock(&running_mutex);
    running_threads = 0;
    pthread_mutex_unlock(&running_mutex);
    return -1.0;
  }
  pthread_mutex_lock(&running_mutex);
  running_threads++;
  pthread_mutex_unlock(&running_mutex);
  if(pthread_create(&gpu_thread, NULL, gpu_filter_thread, NULL) != 0)
  {
    perror("create gpu_thread");
    if(pthread_join(cpu_thread, NULL) != 0)		// try to wait cpu thread end work
    perror("join cpu_thread");
    pthread_mutex_lock(&running_mutex);
    running_threads = 0;
    pthread_mutex_unlock(&running_mutex);
    return -1.0;
  } 
  while (running_threads > 0)				// wait for cpu and gpu threads
  {
    sleep(1);
  }    
  return ((gpu_time)/cpu_time);
}

int main(int argc, char *argv[])
{
  if (argc != 4)
  {
    fprintf(stdout, "%s", ARG_ERROR_MESS);
    return -1;
  }
  #ifdef _OPENMP
    printf("openMP is enabled, dynamic threads count: %d\n", omp_get_num_threads());
  #endif
  unsigned long width_height_backup, step;
  step = atoi(argv[3]);
  width_height_backup = atoi(argv[2]);  
  printf("step: %ld, max: %ld\n", step,  width_height_backup);
  if(width_height_backup <= step)
  {
    printf("you need to specify <image_height-width> > %ld (you specified %ld)", step, width_height_backup);
    return -1;
  }
  //printf("malloc start\n");
  cpu_image = cpu_image_result = gpu_image = NULL;  
  fake_new_wind_image(&cpu_image, width_height_backup, width_height_backup);
  fake_new_wind_image(&cpu_image_result, width_height_backup, width_height_backup);
  fake_new_wind_image(&gpu_image, width_height_backup, width_height_backup);
  unsigned long tmp_ind, tmp_width_height, pixel_count;
  //printf("malloc end\n");
  char *fileBDbin = (char*)malloc(sizeof(char) * FILE_NAME_MAX_LEN);
  char *fileBDtxt = (char*)malloc(sizeof(char) * FILE_NAME_MAX_LEN);
  strcpy(fileBDbin, argv[1]);    
  strcpy(fileBDtxt, argv[1]);
  strcat(fileBDbin, BIN_EXTENSION);
  strcat(fileBDtxt, TXT_EXTENSION);
  
  
  FILE *f_bin = NULL, *f_txt = NULL;
  if((f_bin = fopen(fileBDbin, "wb")) == NULL)			// create file BD bin
  {
    fprintf(stderr, "error open file %s", fileBDbin);
    perror(": ");
    return -1;
  }
  if((f_txt = fopen(fileBDtxt, "wb")) == NULL)			// create file BD txt
  {
    fprintf(stderr, "error open file %s", fileBDtxt);
    perror(":");
    return -1;
  }  
  tmp_width_height = step;
  unsigned long first_step = step;
  while(tmp_width_height <= width_height_backup)
  {   
    cpu_image->height = gpu_image->height = cpu_image->width = gpu_image->width = tmp_width_height;
    pixel_count = (cpu_image->height * cpu_image->width);
    multiplier = calculate_multiplier(argv[1]); 		// calculate_multiplier == gpu_time/cpu_time   
    if(tmp_width_height > 10*step)
      step += (tmp_width_height - step)/5;
    printf("%ldx%ld %ld pixels %f\n", cpu_image->height, cpu_image->width, pixel_count, multiplier);
    fprintf(f_txt, "%ldx%ld %ld pixels ", cpu_image->height,	// fprintf resolutions to file BD txt
	  cpu_image->width, pixel_count);	
    fprintf(f_txt, "%f\n", multiplier);				// fprintf multiplier to file BD txt
    fwrite(&(cpu_image->height) , sizeof(unsigned long), 1, f_bin);	// fwrite height to file BD bin
    fwrite(&multiplier , sizeof(double), 1, f_bin);		// fwrite multiplier to file BD bin
    tmp_width_height += step;
  }
  fseek(f_txt, -1, SEEK_CUR);
  ftruncate(fileno(f_txt), ftell(f_txt));			// crop last byte of file BD
  fclose(f_txt);	
  fclose(f_bin);  
  cpu_image->height = cpu_image_result->height = cpu_image_result->width =
  gpu_image->height = cpu_image->width = gpu_image->width = width_height_backup;  
  free_image(&cpu_image);
  free_image(&gpu_image);
  free_image(&cpu_image_result);
  return 0;
}

