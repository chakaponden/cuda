/*
 * [RUN]:
 * mpiexec -np <X> -hostfile <myHostFile> ./mpiFilter <image_path> <image_filelistname> <mode> 
 * <X>: process count
 * <mode>:
 * -cpu[s] == cpu filter
 * -shared[s] == cuda shared memory filter
 * -async[s] == cuda async filter
 * [s] - if last symbol is 's', then save result to file
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "filters.h"
#define IMAGE_NAME_MAX_LEN 256
#define ARG_ERROR_MESS "mpiexec -np <X> -hostfile <myHostFile> ./mpiFilter <image_path> <image_filelistname> <mode>\n<X>: process count\n<mode>:\n-cpu == cpu filter\n-shared == cuda shared memory filter\n"
#define RESULT_FOLDER "result"
char fileName[IMAGE_NAME_MAX_LEN];

double start_mpi_filter(MPI_Comm comm, int rootRank, char **fileNames, char *filePath, char *mode)
{
  int tmpRank, tmpSize;  
  MPI_Comm_rank(comm, &tmpRank);
  MPI_Comm_size(comm, &tmpSize);  
  MPI_Scatter((*fileNames), IMAGE_NAME_MAX_LEN, MPI_CHAR,
	     fileName, IMAGE_NAME_MAX_LEN, MPI_CHAR,
	     rootRank, comm);
  double execution_time;
  char full_fileName[2*IMAGE_NAME_MAX_LEN];
  strcpy(full_fileName, filePath);
  if(full_fileName[strlen(filePath)-1] != '/')
  {
    full_fileName[strlen(filePath)] = '/';
    full_fileName[strlen(filePath)+1] = '\0';
  }
  strcat(full_fileName, fileName);
  //fprintf(stdout, "src_image [%d]: %s\n", tmpRank, full_fileName);
  wind_image *src_image = NULL, *result_image = NULL;  
  if(new_image(full_fileName, &src_image) < 0)
    return -1.0;
  
/*
 * call filter
 */
  switch(mode[1])
  {
    case 's':					// cuda shared memory filter
    {
      execution_time = cuda_shared_memory(&src_image);
      break;
    }
    case 'a':
    {
      execution_time = async_cuda_filter(&src_image);
      break;
    }
    default:					// default is cpu filter
    {
      execution_time = cpu_filter(&src_image, &result_image);
      break;
    }
  }
  
/*
 * 
 */  
  strcpy(full_fileName, filePath);
  if(full_fileName[strlen(filePath)-1] != '/')
  {
    full_fileName[strlen(filePath)] = '/';
    full_fileName[strlen(filePath)+1] = '\0';
  }
  strcat(full_fileName, RESULT_FOLDER);  
  full_fileName[strlen(filePath)+strlen(RESULT_FOLDER)+1] = '\0';
  if(full_fileName[strlen(filePath)+strlen(RESULT_FOLDER)] != '/')
  {
    full_fileName[strlen(filePath)+strlen(RESULT_FOLDER)+1] = '/';
    full_fileName[strlen(filePath)+strlen(RESULT_FOLDER)+2] = '\0';
  }
  strcat(full_fileName, fileName);
  //fprintf(stdout, "\nresult image [%d]: %s\n", tmpRank, full_fileName);
  if(mode[strlen(mode)-1] == 's')		// save result to file
  {
    switch(mode[1])
    {
      case 's':					// cuda shared memory filter or cuda async filter
      case 'a':
      {
	save_image(full_fileName, &src_image);
	break;
      }
      default:					// default is cpu filter
      {
	save_image(full_fileName, &result_image);
	break;
      }
    }
  }
  free_image(&src_image);
  free_image(&result_image);
  return execution_time;
}

long init_filenames(char *image_filename, char **filenames, long filenames_count, char *filePath)
{
  FILE *f;
  if((f = fopen(image_filename, "rb")) == NULL)
  {
    fprintf(stderr, "error open file %s", image_filename);
    perror(": ");
    return -1;
  }
  (*filenames) = (char*)malloc(filenames_count * sizeof(char) * IMAGE_NAME_MAX_LEN);
  long i, offset, filePathLen;
  filePathLen = strlen(filePath);
  for(i = 0; i < filenames_count; i++)
  {
    offset = i * IMAGE_NAME_MAX_LEN;
    fscanf(f, "%s", ((*filenames)+offset));
  }
  fclose(f);
  return filenames_count;
}

int globalRank, globalSize;

int main(int argc, char *argv[])
{   
    if (argc != 4)
    {
      fprintf(stdout, "%s", ARG_ERROR_MESS);
      return -1;
    }
    long files_count;
    char *filenames;
    double execution_time;
    MPI_Init(&argc, &argv);   
    MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
    MPI_Comm_size(MPI_COMM_WORLD, &globalSize);
    if(globalRank == 0)
    {
      if((files_count = init_filenames(argv[2], &filenames, globalSize, argv[1])) < 0)
      {
	MPI_Abort(MPI_COMM_WORLD, -1);
	return -1;
      }
      /*
      long i, offset;
      for(i = 0; i < files_count; i++)
      {
	offset = i * IMAGE_NAME_MAX_LEN;
	fprintf(stdout, "%s\n", (filenames+offset));
      }
      */
    }
    execution_time = start_mpi_filter(MPI_COMM_WORLD, 0, &filenames, argv[1], argv[3]);
    if(globalRank == 0)
    {
      free(filenames);
    }
    char cpuName[IMAGE_NAME_MAX_LEN];
    int cpuHostNameLen;
    MPI_Get_processor_name(cpuName, &cpuHostNameLen);
    fprintf(stdout, "[%d] %f ms %s %s\n", globalRank, execution_time, cpuName, fileName);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

