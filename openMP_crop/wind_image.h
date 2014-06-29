#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wand/magick_wand.h>
#include "struct_wind_image.h"

#define MAX_PIXEL_VALUE 255
#define FILE_NAME_MAX_LEN 256



#define ThrowWandException(wand) \
{ \
        char *description; \
        ExceptionType severity; \
        description=MagickGetException(wand,&severity); \
        printf("\n\n-----\n%s %s %lu %s\n",GetMagickModule(),description); \
        description=(char *) MagickRelinquishMemory(description); \
        exit(-1); \
}

int new_image(char *image_full_name, wind_image **img)
{
  free_image(img);
  (*img) = (wind_image*)malloc(sizeof(wind_image));
  FILE *f = NULL;
  if((f = fopen(image_full_name, "rb")) == NULL)		// open image file descriptor
  {
    fprintf(stderr, "error open file %s", image_full_name);
    perror(": ");
    return -1;
  }
  MagickWandGenesis();						// initial MagikWand lib
  MagickWand *mw = NULL;					// image object
  mw = NewMagickWand();						// create a wand  
  if(MagickReadImageFile(mw, f) != MagickFalse)			// read image from open descriptor
  {
    (*img)->height = MagickGetImageHeight(mw);
    (*img)->width = MagickGetImageWidth(mw);			// allocate memory for RGB channels
    (*img)->arrayR = (pixel_channel*)malloc((*img)->width * (*img)->height * sizeof(pixel_channel));
    (*img)->arrayG = (pixel_channel*)malloc((*img)->width * (*img)->height * sizeof(pixel_channel));
    (*img)->arrayB = (pixel_channel*)malloc((*img)->width * (*img)->height * sizeof(pixel_channel));
  } 
  else
  {
    mw = DestroyMagickWand(mw);					// free memory magick_wand object
    fclose(f);
    ThrowWandException(mw);
  }  
  resolution ind_height, ind_width, tmp_width;
  PixelIterator *iterator = NULL;
  if((iterator = NewPixelIterator(mw)) == NULL)
  {   
    mw = DestroyMagickWand(mw);					// free memory magick_wand object
    fclose(f);
    ThrowWandException(mw);    
  }
  PixelWand **pixels = NULL;
  for(ind_height = 0; ind_height < (*img)->height; ind_height++)// get RGB channels for each pixel
  {
    pixels = PixelGetNextIteratorRow(iterator, &tmp_width);	// get row pixels
    for(ind_width = 0; ind_width < (*img)->width; ind_width++)	// get RGB channels each pixel in row
    {
      (*img)->arrayR[ind_width + (*img)->width * ind_height] = (pixel_channel)(MAX_PIXEL_VALUE * PixelGetRed(pixels[ind_width]));
      (*img)->arrayG[ind_width + (*img)->width * ind_height] = (pixel_channel)(MAX_PIXEL_VALUE * PixelGetGreen(pixels[ind_width]));
      (*img)->arrayB[ind_width + (*img)->width * ind_height] = (pixel_channel)(MAX_PIXEL_VALUE * PixelGetBlue(pixels[ind_width]));
    }  
  }  
  iterator = DestroyPixelIterator(iterator);			// free memory magick_wand iterator
  mw = DestroyMagickWand(mw);					// free memory magick_wand object
  MagickWandTerminus();						// end work with MagikWand lib
  fclose(f);							// close file descriptor  
  return 0;
}

int save_image(char *image_full_name, wind_image **img)
{
  MagickWandGenesis();						// initial MagikWand lib
  MagickWand *mw = NULL;					// image object
  mw = NewMagickWand();						// create a wand
  PixelWand *tmp_pixel = NewPixelWand();
  PixelSetColor(tmp_pixel, "rgb(255,255,255)");			// set white pixel
  
  if(MagickNewImage(mw, (*img)->width, (*img)->height, tmp_pixel) == MagickFalse)
  {
    mw = DestroyMagickWand(mw);					// free memory magick_wand object
    tmp_pixel = DestroyPixelWand(tmp_pixel);
    ThrowWandException(mw);
  }
  PixelIterator *iterator = NULL;  
  if((iterator = NewPixelIterator(mw)) == NULL)
  {   
    mw = DestroyMagickWand(mw);					// free memory magick_wand object
    tmp_pixel = DestroyPixelWand(tmp_pixel);
    ThrowWandException(mw);    
  }  
  PixelWand **pixels = NULL;
  resolution ind_height, ind_width, tmp_width;  
  for(ind_height = 0; ind_height < (*img)->height; ind_height++)// get RGB channels for each pixel
  {
    pixels = PixelGetNextIteratorRow(iterator, &tmp_width);	// get row pixels
    for(ind_width = 0; ind_width < (*img)->width; ind_width++)	// get RGB channels each pixel in row
    {
      PixelSetRed(tmp_pixel,   ((double)((unsigned char)(*img)->arrayR[ind_width + (*img)->width * ind_height]))/MAX_PIXEL_VALUE);
      PixelSetGreen(tmp_pixel, ((double)((unsigned char)(*img)->arrayG[ind_width + (*img)->width * ind_height]))/MAX_PIXEL_VALUE);
      PixelSetBlue(tmp_pixel,  ((double)((unsigned char)(*img)->arrayB[ind_width + (*img)->width * ind_height]))/MAX_PIXEL_VALUE);
      pixels[ind_width] = ClonePixelWand(tmp_pixel);     
    }
    PixelSyncIterator(iterator);
  }    
  if(MagickWriteImage(mw, image_full_name) == MagickFalse)
  {
    tmp_pixel = DestroyPixelWand(tmp_pixel);
    mw = DestroyMagickWand(mw);					// free memory magick_wand object
    iterator = DestroyPixelIterator(iterator);
    ThrowWandException(mw);
  }
  /*
  if(((*img)->height * (*img)->width) <= 1)			// if image resolution <= 1024x768
  {								// then free MagikWand resource object
    printf("freeMem start\n");					// there are bugs if free MagikWand memory objects with
    tmp_pixel = DestroyPixelWand(tmp_pixel);			// more 1024x768 resolution
    iterator = DestroyPixelIterator(iterator);	
    mw = DestroyMagickWand(mw);					// free memory magick_wand object
    MagickWandTerminus();					// end work with MagikWand lib
    printf("freeMem end\n");
  }
  */
  return 0;
}

// return files count, allocate memory char **filenames
long get_folder_files(char *file_path, char ***filenames, double **image_resolutions, double **image_height)
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
    (*image_height) = (double*)malloc(files_count * sizeof(double));
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
      (*image_height)[files_count] = (double)imageH;
      files_count++;
      fclose(f);
    }
  }
  closedir(d);
  mw = DestroyMagickWand(mw);			// free memory
  MagickWandTerminus();				// end work with MagikWand lib
  return files_count;
}

void free_image_info(char ***filenames, double **image_resolutions, double **image_height, long files_count)
{
  long i;
  for(i = 0; i < files_count; i++)
    free((*filenames)[i]);
  free((*filenames));
  free((*image_resolutions));
  free((*image_height));
}