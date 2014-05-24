/*
 * [COPMILE]:
 * mpicc -o mpiFilter mpiFilter.c -std=gnu99 -lm
 * 
 * [RUN]:
 * mpiexec -np <X> -hostfile <myHostFile> ./mpiFilter <image_path> <image_filelistname>\n<X> - process count\n
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define IMAGE_NAME_MAX_LEN 256
#define ARG_ERROR_MESS "mpiexec -np <X> -hostfile <myHostFile> ./mpiFilter <image_path> <image_filelistname>\n<X> - process count\n"

char fileName[IMAGE_NAME_MAX_LEN];

int funcForBaranov(MPI_Comm comm, int rootRank, char **fileNames)
{
  int tmpRank, tmpSize;  
  MPI_Comm_rank(comm, &tmpRank);
  MPI_Comm_size(comm, &tmpSize);  
  MPI_Scatter((*fileNames), IMAGE_NAME_MAX_LEN, MPI_CHAR,
	     fileName, IMAGE_NAME_MAX_LEN, MPI_CHAR,
	     rootRank, comm);
  
  /*
   * 
   * need to insert filter image code here with filename char fileName[CPU_NAME_MAX_LEN]
   * 
   */  
  
  return tmpSize;
}

long init_filenames(char *image_filename, char **filenames, long filenames_count)
{
  FILE *f;
  if((f = fopen(image_filename, "rb")) == NULL)
  {
    fprintf(stderr, "error open file %s", image_filename);
    perror(": ");
    return -1;
  }
  (*filenames) = (char*)malloc(filenames_count * sizeof(char) * IMAGE_NAME_MAX_LEN);
  long i, offset;
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
    if (argc != 3)
    {
      fprintf(stdout, "%s", ARG_ERROR_MESS);
      return -1;
    }
    char *filenames;
    long files_count;
    double starttime, endtime;
    MPI_Init(&argc, &argv);   
    MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
    MPI_Comm_size(MPI_COMM_WORLD, &globalSize);
    if(globalRank == 0)
    {
      if((files_count = init_filenames(argv[2], &filenames, globalSize)) < 0)
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
    starttime = MPI_Wtime();
    files_count = funcForBaranov(MPI_COMM_WORLD, 0, &filenames);
    endtime   = MPI_Wtime();
    if(globalRank == 0)
    {
      free(filenames);
    }
    char cpuName[IMAGE_NAME_MAX_LEN];
    int cpuHostNameLen;
    MPI_Get_processor_name(cpuName, &cpuHostNameLen);
    fprintf(stdout, "[%s][%d]: %f %s\n", cpuName, globalRank, endtime-starttime, fileName);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

