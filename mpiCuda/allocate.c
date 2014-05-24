/*
 * [COMPILE]:
 * cc `MagickWand-config --cflags --cppflags` -o startMpiFilter allocate.c `MagickWand-config --ldflags --libs`
 * 
 * [RUN]:
 * ./startMpiFilter <file_path> [hosts_ipv4]
 * 
 */

#include <dirent.h>
#include <wand/magick_wand.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARG_ERROR_MESS	"[RUN]:\n./startMpiFilter <images_path> [hosts_ipv4]\n"
#define HOSTS_FILE_NAME "hosts.txt"
#define IMAGES_FILE_NAME "images.txt"
#define MPI_RUN_COMMAND "mpiexec"
#define MPI_PROGRAM_FILE "mpiFilter"


#define ThrowWandException(wand) \
{ \
        char *description; \
        ExceptionType severity; \
        description=MagickGetException(wand,&severity); \
        printf("\n\n-----\n%s %s %lu %s\n",GetMagickModule(),description); \
        description=(char *) MagickRelinquishMemory(description); \
        exit(-1); \
}


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
  MagickWand *mw = NULL;			// image object
  size_t imageH, imageW;
  char imageFullName[1024];  
  strcpy(imageFullName, file_path);  
  if(imageFullName[strlen(file_path)-1] != '/')
  {
    imageFullName[strlen(file_path)] = '/';
    imageFullName[strlen(file_path)+1] = '\0';
  }
  long pathLen = strlen(imageFullName);
  FILE *f = NULL;
  MagickWandGenesis();				// initial MagikWand lib
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
  long images_num, hosts_num, ind_tmp;
  char **filenames;
  double *image_resolutions, *host_resolutions;
  long ind_max_image, ind_max_host;
  double max_image_resolution, max_host_resolution;
  long double all_resolutions = 0; 
  
/*
 * load image filenames and it's number of pixels to filenames and image_resolutions from argv[1] path
 */
  
  if((images_num = get_folder_files(argv[1], &filenames, &image_resolutions)) > 0)
  {    
/*
 *
 */    
    hosts_num = argc - 2;
    host_resolutions = (double*)malloc(hosts_num * sizeof(double));
    for(ind_tmp = 0; ind_tmp < images_num; ind_tmp++)
      all_resolutions += image_resolutions[ind_tmp];
    all_resolutions = all_resolutions/hosts_num;    
    for(ind_tmp = 0; ind_tmp < hosts_num; ind_tmp++)
      host_resolutions[ind_tmp] = (double)all_resolutions;
    
    // show each images name and it's number of pixels
    /*
    for(ind_tmp = 0; ind_tmp < images_num; ind_tmp++)
      printf("%s: %0.0f\n", filenames[ind_tmp], image_resolutions[ind_tmp]);
    */
    
    
/*
 * associate images to hosts
 */
    do
    {
      max_host_resolution = host_resolutions[0];
      ind_max_host = 0;
      max_image_resolution = 0;      
      for(ind_tmp = 0; ind_tmp < hosts_num; ind_tmp++)			// find host with max free available pixels
      {
	if(max_host_resolution < host_resolutions[ind_tmp])
	{
	  max_host_resolution = host_resolutions[ind_tmp];
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
	host_resolutions[ind_max_host] -= max_image_resolution;
      }      
    }while(max_image_resolution > 0);
    
/*
 *
 */

/*
 * write images names to file according associated host number (from low to high host number)
 * and create hostsFile for mpi
 */
    long ind_host, host_images_num;
    double exchange_tmp;
    FILE *f_hosts, *f_images;
    if((f_hosts = fopen(HOSTS_FILE_NAME, "wb")) == NULL)
    {
      fprintf(stderr, "error open file %s", HOSTS_FILE_NAME);
      perror(": ");
      return -1;
    }
    if((f_images = fopen(IMAGES_FILE_NAME, "wb")) == NULL)
    {
      fprintf(stderr, "error open file %s", IMAGES_FILE_NAME);
      perror(": ");
      return -1;
    }
    for(ind_host = -1; ind_host > -hosts_num-1; ind_host--)
    {
      fprintf(f_hosts, "%s slots=", argv[(-ind_host)+1]);		// fprintf host_ipv4 to file hosts
      host_images_num = 0;
      for(ind_tmp = 0; ind_tmp < images_num; ind_tmp++)
      {
	if(image_resolutions[ind_tmp] == (double)ind_host)
	{
	  host_images_num++;
	  fprintf(f_images, "%s\n", filenames[ind_tmp]);		// fprintf image_name to file imsges
	}
	  
      }
      fprintf(f_hosts, "%ld\n", host_images_num);			// fprintf host_images_num to file hosts
    }
    fseek(f_hosts, -1, SEEK_CUR);
    ftruncate(fileno(f_hosts), ftell(f_hosts));				// crop last byte of file hosts
    fclose(f_hosts);
    fseek(f_images, -1, SEEK_CUR);
    ftruncate(fileno(f_images), ftell(f_images));			// crop last byte of file images
    fclose(f_images);
    
/*
 *
 */
   

  }
  else
    return (int)images_num;
  free_image_info(&filenames, &image_resolutions, images_num);
  
  char proc_num[32];
  sprintf(proc_num, "%ld", images_num);
  char *argList[] = { "mpiexec", "-np", proc_num, "-hostfile", HOSTS_FILE_NAME, MPI_PROGRAM_FILE, argv[1], IMAGES_FILE_NAME, NULL };
  execvp("mpiexec", argList);
  return 0;
}

