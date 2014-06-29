/*
 * [COMPILE]:
 * gcc -c openMP.c -o openMP.o `pkg-config --cflags --libs MagickWand`
 * 
 * [RUN]:
 * ./openMP </full_path/images_path> </full_path/BD_file> [-s]
 * [-s] - if exist, then save result to file
 * 
 */

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "cuda_crop_wind_image.h"
#include "filters.h"
#define RESULT_FOLDER "result"

#define ARG_ERROR_MESS	"[RUN]:\n./openMP </full_path/images_path> </full_path/BD_file> [-s]\n[-s] - if exist, then save result to file\n"

resolution sizesCount;
resolution *pixelsCount;
double *multiplier;


int init_multipliers(char *full_filepath)
{
  FILE *f = NULL;
  if((f = fopen(full_filepath, "rb")) == NULL)		// open BD file
  {
    fprintf(stderr, "error open file %s", full_filepath);
    perror(": ");
    return -1;
  }
  sizesCount = 0;
  unsigned long step_bytes = sizeof(resolution) + sizeof(double);
  while(fgetc(f) != EOF)
  {
    fseek(f, -(sizeof(char)), SEEK_CUR);
    fseek(f, step_bytes, SEEK_CUR);
    sizesCount++;    
  }
  rewind(f);
  fseek(f, 0, SEEK_SET);
  resolution ind_tmp;
  pixelsCount = (resolution*)malloc(sizeof(resolution) * sizesCount);
  multiplier = (double*)malloc(sizeof(double) * sizesCount);
  for(ind_tmp = 0; ind_tmp < sizesCount; ind_tmp++)
  {
    fread((void*)(&pixelsCount[ind_tmp]), sizeof(resolution), 1, f);
    fread((void*)(&multiplier[ind_tmp	]), sizeof(double), 1, f);  
  }
  fclose(f);  
  return 0;  
}


double get_multiplier(resolution local_pixelsCount, resolution sizes, resolution *pixels, double *multipl)
{
  resolution ind_tmp;
  for(ind_tmp = 0; ind_tmp < sizes; ind_tmp++)
  {
    if(local_pixelsCount <= pixels[ind_tmp])
    {
      if(local_pixelsCount < pixels[ind_tmp])
      {
	if(ind_tmp != 0)
	  return (multipl[ind_tmp]+multipl[ind_tmp-1])/2;
      }
      return multipl[ind_tmp];
    }
  }
  return multipl[sizes-1];
}

int start_filter(char **fileNames, char *filePath, char mode, long ind)
{
  char *fileName = fileNames[ind];
  char full_fileName[2*FILE_NAME_MAX_LEN];
  strcpy(full_fileName, filePath);
  strcat(full_fileName, fileName);    
  if(new_cuda_crop_image(full_fileName, &cpu_image, &gpu_image, 
			  sizesCount, pixelsCount, multiplier) < 0)	// read image and init image RGB channels
    return -1;
  gpu_time = -1.0;  cpu_time = 1.0; running_threads = 0;
/*
 * call filters
 */
    pthread_mutex_lock(&running_mutex);
    running_threads++;
    pthread_mutex_unlock(&running_mutex);
    if(pthread_create(&cpu_thread, NULL, cpu_filter_thread, NULL) != 0)
    {
      perror("create cpu_thread");
      free_image(&cpu_image);				// free cpu_image
      pthread_mutex_lock(&running_mutex);
      running_threads = 0;
      pthread_mutex_unlock(&running_mutex);
      return -1;
    }
    pthread_mutex_lock(&running_mutex);
    running_threads++;
    pthread_mutex_unlock(&running_mutex);
    if(pthread_create(&gpu_thread, NULL, gpu_filter_thread, NULL) != 0)
    {
      perror("create gpu_thread");
      if(pthread_join(cpu_thread, NULL) != 0)		// try to wait cpu thread end work
	perror("join cpu_thread");
      free_image(&cpu_image_result);			// free cpu_result
      free_image(&cpu_image);				// free cpu_image
      free_image(&gpu_image);				// free gpu_image
      pthread_mutex_lock(&running_mutex);
      running_threads = 0;
      pthread_mutex_unlock(&running_mutex);
      return -1;
    }
    while (running_threads > 0)				// wait for cpu and gpu threads
    {
      sleep(1);
    }        
/*
 * 
 */  
  if(mode == 's')					// save image
  {
    strcpy(full_fileName, filePath);
    resolution tmp_len = strlen(filePath);
    full_fileName[tmp_len] = '\0';
    if(full_fileName[tmp_len-1] != '/')
    {
      full_fileName[tmp_len] = '/';
      full_fileName[tmp_len+1] = '\0';
      tmp_len++;
    }
    strcat(full_fileName, RESULT_FOLDER);  
    tmp_len +=  strlen(RESULT_FOLDER);
    if(full_fileName[tmp_len-1] != '/')
    {
      full_fileName[tmp_len] = '/';
      full_fileName[tmp_len+1] = '\0';
      tmp_len++;
    }
    strcat(full_fileName, fileName);
    return save_cuda_crop_bmp(full_fileName, &cpu_image_result, &gpu_image, result_bmp);
    //return save_cuda_crop_image(full_fileName, &cpu_image_result, &gpu_image);
  }
  return 0;
}



int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    fprintf(stdout, "%s", ARG_ERROR_MESS);
    return -1;
  }
  #ifdef _OPENMP
    printf("openMP is enabled, dynamic threads count: %d\n", omp_get_num_threads());
  #endif
  char c = 'r';
  if(argc == 4 && !strcmp(argv[3], "-s"))
    c = 's';
  long images_num, ind_tmp;
  char **filenames;
  double *image_resolutions; 
  double *image_height;
  
/*
 * load image filenames and it's number of pixels to filenames and image_resolutions from argv[1] path
 */
  
  if((images_num = get_folder_files(argv[1], &filenames, &image_resolutions, &image_height)) > 0)
  {    
/*
 *
 */    

    if(init_multipliers(argv[2]) != 0)
    {
      free_image_info(&filenames, &image_resolutions, &image_height, images_num);
      return -1;
    }
    char right_file_path[FILE_NAME_MAX_LEN];
    strcpy(right_file_path, argv[1]);
    if(right_file_path[strlen(argv[1])-1] != '/')
    {
      right_file_path[strlen(argv[1])] = '/';
      right_file_path[strlen(argv[1])+1] = '\0';
    }
    double max_resolution = 0.0, max_height = 0.0;
    for(ind_tmp = 0; ind_tmp < images_num; ind_tmp++)
    {
      if(image_resolutions[ind_tmp] > max_resolution)
      {
	max_resolution = image_resolutions[ind_tmp];      
	max_height = image_height[ind_tmp];
      }
    }
    printf("max_image: %0.0fx%0.0f\n", max_height, max_resolution/max_height);
    cpu_image = cpu_image_result = gpu_image = NULL;
    gpu_image = (wind_image*)malloc(sizeof(wind_image) * 1);
    fake_new_wind_image(&cpu_image, (resolution)(max_height+2), (resolution)(max_resolution/max_height));
    fake_new_wind_image(&cpu_image_result, (resolution)(max_height+2), (resolution)(max_resolution/max_height));
    result_bmp = (unsigned char *)malloc(3 * (resolution)max_resolution * sizeof(unsigned char));
    
/*
 * filter image by cpu and gpu in parallel, buy every image in order (one image at one time)
 */
    printf("-------------------------------------------------------------------\n");
    printf("multipliers:\n");
    for(ind_tmp = 0; ind_tmp < sizesCount; ind_tmp++)
      printf("%ld pixels %f\n", pixelsCount[ind_tmp], multiplier[ind_tmp]);
    printf("-------------------------------------------------------------------\n");
    /*
    printf("files:\n");
    for(ind_tmp = 0; ind_tmp < images_num; ind_tmp++)
     printf("%0.0f pixels %s\n", image_resolutions[ind_tmp], filenames[ind_tmp]); 
    printf("-------------------------------------------------------------------\n");
    */
    MagickWandGenesis();						// initial MagikWand lib
    for(ind_tmp = 0; ind_tmp < images_num; ind_tmp++)
    {
      if(!start_filter(filenames, right_file_path, c, ind_tmp))		// if filter ok
      {      
	printf("GPU: %f ms %s\n", gpu_time, filenames[ind_tmp]);
	printf("CPU: %f ms %s\n", cpu_time, filenames[ind_tmp]);    
	printf("-------------------------------------------------------------------\n");
      }
      else								// if error
      {
	MagickWandTerminus();						// end work with MagikWand lib
	free(result_bmp);
	free(pixelsCount);
	free(multiplier);
	free(gpu_image);
	free_image(&cpu_image_result);			// free cpu_result
	free_image(&cpu_image);
	free_image_info(&filenames, &image_resolutions, &image_height, images_num);
	return -1;
      }
    }
    MagickWandTerminus();						// end work with MagikWand lib
    
/*
 *
 */
    free(result_bmp);
    free(gpu_image);
    free_image(&cpu_image_result);			// free cpu_result
    free_image(&cpu_image);
    free(pixelsCount);
    free(multiplier);
    free_image_info(&filenames, &image_resolutions, &image_height, images_num);
    return 0;
  }
  else
    return (int)images_num;
}

