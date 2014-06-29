/*
 * [COMPILE]:
 * mpicc -c openMP.c -o openMP.o `Wand-config --cflags`
 * 
 */

#include <dirent.h>
#include <wand/magick_wand.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ARG_ERROR_MESS	"[RUN]:\n./openMP <images_path> <testImage> [-s]\n[-s] - if exist, then save result to file\n"

#define ThrowWandException(wand) \
{ \
        char *description; \
        ExceptionType severity; \
        description=MagickGetException(wand,&severity); \
        printf("\n\n-----\n%s %s %lu %s\n",GetMagickModule(),description); \
        description=(char *) MagickRelinquishMemory(description); \
        exit(-1); \
}

#include "filters.h"

// return files count, allocate memory char **filenames
long get_folder_files(char *file_path, char ***filenames, double **image_resolutions)
{
  DIR *d;
  struct dirent *dir;
  if(!(d = opendir(file_path)))
  {
    perror("error: open images directory files count: ");
    return -1;
  }
  long files_count = 0;
  while((dir = readdir(d)) != NULL)		// calculate files count at images file path
  {
    if(dir->d_type == DT_REG)
      files_count++;
  }
  if(files_count)
  {
    (*filenames) = (char**)malloc(files_count * sizeof(char*));
    (*image_resolutions) = (double*)malloc(files_count * sizeof(double));
  }
  else
  {
    fprintf(stdout, "error: no files at '%s'", file_path);
    closedir(d);
    return 0;
  }
  rewinddir(d);
  files_count = 0;
  size_t imageH, imageW;
  char imageFullName[2 * FILE_NAME_MAX_LEN];  
  strcpy(imageFullName, file_path);  
  if(imageFullName[strlen(file_path)-1] != '/')
  {
    imageFullName[strlen(file_path)] = '/';
    imageFullName[strlen(file_path)+1] = '\0';
  }
  long pathLen = strlen(imageFullName);
  FILE *f = NULL;
  MagickWandGenesis();				// initial MagikWand lib
  MagickWand *mw = NULL;			// image object
  mw = NewMagickWand();				// create a wand
  while((dir = readdir(d)) != NULL)		// get name of each file at image file path
  {
    if(dir->d_type == DT_REG)
    {
      strcat(imageFullName, dir->d_name);
      if((f = fopen(imageFullName, "rb")) == NULL)
      {
	fprintf(stderr, "error open file %s", imageFullName);
	perror(": ");
	return -1;
      }
      imageH = imageW = 0;
						// if no error read image then get image height and width           
      if(MagickReadImageFile(mw, f) != MagickFalse)
      {
	imageH = MagickGetImageHeight(mw);
	imageW = MagickGetImageWidth(mw);
      }
      else
	ThrowWandException(mw);
      imageFullName[pathLen] = '\0';
      (*filenames)[files_count] = (char*)malloc(dir->d_reclen * sizeof(char));
      strcpy((*filenames)[files_count], dir->d_name);
      (*image_resolutions)[files_count] = (double)imageH * (double)imageW;
      files_count++;
      fclose(f);
    }
  }
  closedir(d);
  mw = DestroyMagickWand(mw);			// free memory
  MagickWandTerminus();				// end work with MagikWand lib
  return files_count;
}

void free_image_info(char ***filenames, double **image_resolutions, long files_count)
{
  long i;
  for(i = 0; i < files_count; i++)
    free((*filenames)[i]);
  free((*filenames));
  free((*image_resolutions));
}

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    fprintf(stdout, "%s", ARG_ERROR_MESS);
    return -1;
  }
  char c = 'r';
  if(argc == 4 && !strcmp(argv[3], "-s"))
    c = 's';
  long images_num, hosts_num, ind_tmp;
  char **filenames;
  double *image_resolutions, *quota;
  long ind_max_image, ind_max_host;
  double max_image_resolution, max_host_resolution;
  long double all_resolutions = 0, multiplier = 1; 
  char **testImage = (char**)malloc(sizeof(char*) * 1);
  testImage[0] = (char*)malloc(sizeof(char) * FILE_NAME_MAX_LEN);
  strcpy(testImage[0], argv[2]);
  
  clock_t begin, end;  	  
  double cpu_time, gpu_time;
  begin = clock();
  start_filter(testImage, argv[1], 'r', -1, 0);
  end = clock();  
  cpu_time = ((double)(end - begin) / (CLOCKS_PER_SEC/1000));
  printf("test CPU: %f ms %s", cpu_time, argv[2]);
  begin = clock();
  start_filter(testImage, argv[1], 'r', -2, 0);
  end = clock();
  gpu_time = ((double)(end - begin) / (CLOCKS_PER_SEC/1000));
  printf("test GPU: %f ms %s", gpu_time, argv[2]);
  multiplier = cpu_time/gpu_time;
  
/*
 * load image filenames and it's number of pixels to filenames and image_resolutions from argv[1] path
 */
  
  if((images_num = get_folder_files(argv[1], &filenames, &image_resolutions)) > 0)
  {    
/*
 *
 */    
    hosts_num = 2;
    quota = (double*)malloc(hosts_num * sizeof(double));
    for(ind_tmp = 0; ind_tmp < images_num; ind_tmp++)
      all_resolutions += image_resolutions[ind_tmp];
    all_resolutions = all_resolutions/hosts_num;    
    for(ind_tmp = 0; ind_tmp < hosts_num; ind_tmp++)
      quota[ind_tmp] = (double)all_resolutions;		// initial cpu (quota[0]) and gpu (quota[1]) quotas
    quota[1] *= multiplier;				// apply myltiplier to gpu quota   
    
/*
 * associate images to hosts
 */
    do
    {
      max_host_resolution = quota[0];
      ind_max_host = 0;
      max_image_resolution = 0;      
      for(ind_tmp = 0; ind_tmp < hosts_num; ind_tmp++)			// find cpu-gpu with max free available pixels quota
      {
	if(max_host_resolution < quota[ind_tmp])
	{
	  max_host_resolution = quota[ind_tmp];
	  ind_max_host = ind_tmp;
	}
      }
      
      for(ind_tmp = 0; ind_tmp < images_num; ind_tmp++)			// find image with max pixel count
      {
	if(max_image_resolution < image_resolutions[ind_tmp])
	{
	  max_image_resolution = image_resolutions[ind_tmp];
	  ind_max_image = ind_tmp;
	}
      }            
      if(max_image_resolution > 0)
      {
	image_resolutions[ind_max_image] = (double)(-ind_max_host-1);
	quota[ind_max_host] -= max_image_resolution;
      }      
    }while(max_image_resolution > 0);
    
/*
 *
 */

/*
 * -1 == cpu
 * -2 == gpu
 */
    pid_t child[images_num];
    int execution_time[images_num];	
    for(ind_tmp = 0; ind_tmp < images_num; ind_tmp++)
    {
      switch((child[ind_tmp] = fork()))
      {
	case -1:							// fork error
	{
	  perror("fork error, do not wait all child proc ");
	  
	  free_image_info(&filenames, &image_resolutions, images_num);
	  free(quota);	  
	  return -1;
	}
	case 0:								// child
	{
	  begin = clock();	  	  
	  start_filter(filenames, argv[1], c, (int)(image_resolutions[ind_tmp]), ind_tmp);
	  free_image_info(&filenames, &image_resolutions, images_num);
	  free(quota);
	  end = clock();
	  return (int)((double)(end - begin) / (CLOCKS_PER_SEC/1000));
	}
	default:							// parent
	{
	  break;
	}
      }      
    }
    int status, wait_child = 1;
    pid_t childPid;
    while(wait_child)
    {
      switch((childPid = wait(&status))) 
      {						// WNOHANG == return control immediately
	case -1:					// error waitpid
	{
	  perror("waitpid error");
	  return -1;
	}
	case  0:					// no child proc terminated now
	{
	  break;
	}
	default:					// child proc terminated
	{		      
	  for(ind_tmp = 0; ind_tmp < images_num; ind_tmp++)		
	  {
	    if(child[ind_tmp] == childPid)
	    {
	      child[ind_tmp] = -1;				// set pid as -1
	      if (WIFEXITED (status))
		execution_time[ind_tmp] = WEXITSTATUS(status);	// get child return value		
	      break;
	    }		      
	  }
	  wait_child = 0;
	  for(ind_tmp = 0; ind_tmp < images_num; ind_tmp++)
	  {
	    if(child[ind_tmp] != -1)
	    {
	      wait_child = 1;
	      break;
	    }	      
	  }
	  break;
	}		
      }
    }
/*
 *
 */
    for(ind_tmp = 0; ind_tmp < images_num; ind_tmp++)
    {
      if(image_resolutions[ind_tmp] == -1)	// cpu time
	printf("CPU: %d ms %s", execution_time[ind_tmp], filenames[ind_tmp]);
      else					// gpu time
	printf("GPU: %d ms %s", execution_time[ind_tmp], filenames[ind_tmp]);  
    }
    free_image_info(&filenames, &image_resolutions, images_num);
    free(quota);
    return 0;
  }
  else
    return (int)images_num;
}

