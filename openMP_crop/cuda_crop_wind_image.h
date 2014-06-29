#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wand/magick_wand.h>
#include "struct_wind_image.h"

#define BMP_EXTENSION ".bmp"
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


double get_multiplier(resolution local_pixelsCount, resolution sizes, resolution *pixelsCount, double *multipl);

int new_cuda_crop_image(char *image_full_name, wind_image **img0, wind_image **img1, 
			resolution sizes, resolution *pixelsCount, double *multipl)
{
  
  FILE *f = NULL;
  if((f = fopen(image_full_name, "rb")) == NULL)		// open image file descriptor
  {
    fprintf(stderr, "error open file %s", image_full_name);
    perror(": ");
    return -1;
  }
  // MagickWandGenesis();					// initial MagikWand lib
  MagickWand *mw = NULL;					// image object
  mw = NewMagickWand();						// create a wand  
  double current_multipl;
  resolution real_image_height;
  if(MagickReadImageFile(mw, f) != MagickFalse)			// read image from open descriptor
  {
    real_image_height = (*img0)->height = MagickGetImageHeight(mw);
    (*img0)->width = (*img1)->width = MagickGetImageWidth(mw);
    current_multipl = get_multiplier((*img0)->height, sizes, pixelsCount, multipl);
    printf("real %ldx%ld  %f  %s\n", real_image_height, (*img0)->width, current_multipl, image_full_name);
    
    double width_charge = ((double)(*img0)->width/real_image_height);
    width_charge = ((double)1.0)/width_charge;
    
    (*img0)->height += 2;
								// allocate memory for RGB channels    
    (*img0)->height = (double)real_image_height*(current_multipl/(current_multipl + 1.0))*width_charge;
    (*img1)->height = real_image_height - (*img0)->height;
    (*img0)->height++; (*img1)->height++;
    resolution offset = (*img0)->width * (*img0)->height;
    (*img1)->arrayR = ((*img0)->arrayR + offset);
    (*img1)->arrayG = ((*img0)->arrayG + offset);
    (*img1)->arrayB = ((*img0)->arrayB + offset);    
  } 
  else
  {
    mw = DestroyMagickWand(mw);					// free memory magick_wand object
    fclose(f);
    ThrowWandException(mw);
  }  

  printf("cpu: %ldx%ld pointer: %p\ngpu: %ldx%ld pointer: %p\n\n", 
	 (*img0)->height, (*img0)->width, ((*img0)->arrayR), (*img1)->height, (*img1)->width, ((*img1)->arrayR));

  resolution ind_height, ind_width, tmp_width;
  PixelIterator *iterator = NULL;
  if((iterator = NewPixelIterator(mw)) == NULL)
  {   
    mw = DestroyMagickWand(mw);					// free memory magick_wand object    
    fclose(f);
    ThrowWandException(mw);    
  }
  PixelWand **pixels = NULL;
  for(ind_height = 0; ind_height < (*img0)->height; ind_height++)// get RGB channels for each pixel
  {
    pixels = PixelGetNextIteratorRow(iterator, &tmp_width);	// get row pixels
    for(ind_width = 0; ind_width < (*img0)->width; ind_width++)	// get RGB channels each pixel in row
    {
      (*img0)->arrayR[ind_width + (*img0)->width * ind_height] = (pixel_channel)(MAX_PIXEL_VALUE * PixelGetRed(pixels[ind_width]));
      (*img0)->arrayG[ind_width + (*img0)->width * ind_height] = (pixel_channel)(MAX_PIXEL_VALUE * PixelGetGreen(pixels[ind_width]));
      (*img0)->arrayB[ind_width + (*img0)->width * ind_height] = (pixel_channel)(MAX_PIXEL_VALUE * PixelGetBlue(pixels[ind_width]));
    }  
  }  
  pixels = PixelGetPreviousIteratorRow(iterator, &tmp_width);
  for(ind_height = 0; ind_height < (*img1)->height; ind_height++)// get RGB channels for each pixel
  {
    for(ind_width = 0; ind_width < (*img1)->width; ind_width++)	// get RGB channels each pixel in row
    {
      (*img1)->arrayR[ind_width + (*img1)->width * ind_height] = (pixel_channel)(MAX_PIXEL_VALUE * PixelGetRed(pixels[ind_width]));
      (*img1)->arrayG[ind_width + (*img1)->width * ind_height] = (pixel_channel)(MAX_PIXEL_VALUE * PixelGetGreen(pixels[ind_width]));
      (*img1)->arrayB[ind_width + (*img1)->width * ind_height] = (pixel_channel)(MAX_PIXEL_VALUE * PixelGetBlue(pixels[ind_width]));
    }
    pixels = PixelGetNextIteratorRow(iterator, &tmp_width);	// get row pixels
  }  
  iterator = DestroyPixelIterator(iterator);			// free memory magick_wand iterator
  mw = DestroyMagickWand(mw);					// free memory magick_wand object
  //MagickWandTerminus();						// end work with MagikWand lib
  fclose(f);							// close file descriptor  
  return 0;
}

int save_cuda_crop_image(char *image_full_name, wind_image **img0, wind_image **img1)
{
  //MagickWandGenesis();						// initial MagikWand lib
  printf("save to file start\n");
  MagickWand *mw = NULL;					// image object
  mw = NewMagickWand();						// create a wand
  PixelWand *tmp_pixel = NewPixelWand();
  PixelSetColor(tmp_pixel, "rgb(255,255,255)");			// set white pixel
  resolution real_height = ((*img0)->height+(*img1)->height-2);
  if(MagickNewImage(mw, (*img0)->width, real_height, tmp_pixel) == MagickFalse)
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
  for(ind_height = 0; ind_height < (*img0)->height-1; ind_height++)// get RGB channels for each pixel
  {
    pixels = PixelGetNextIteratorRow(iterator, &tmp_width);	// get row pixels
    for(ind_width = 0; ind_width < (*img0)->width; ind_width++)	// get RGB channels each pixel in row
    {
      PixelSetRed(tmp_pixel,   ((double)((unsigned char)(*img0)->arrayR[ind_width + (*img0)->width * ind_height]))/MAX_PIXEL_VALUE);
      PixelSetGreen(tmp_pixel, ((double)((unsigned char)(*img0)->arrayG[ind_width + (*img0)->width * ind_height]))/MAX_PIXEL_VALUE);
      PixelSetBlue(tmp_pixel,  ((double)((unsigned char)(*img0)->arrayB[ind_width + (*img0)->width * ind_height]))/MAX_PIXEL_VALUE);
      pixels[ind_width] = ClonePixelWand(tmp_pixel);     
    }
    PixelSyncIterator(iterator);
  }    
  for(ind_height = 1; ind_height < (*img1)->height; ind_height++)// get RGB channels for each pixel
  {
    pixels = PixelGetNextIteratorRow(iterator, &tmp_width);	// get row pixels
    for(ind_width = 0; ind_width < (*img1)->width; ind_width++)	// get RGB channels each pixel in row
    {
      PixelSetRed(tmp_pixel,   ((double)((unsigned char)(*img1)->arrayR[ind_width + (*img1)->width * ind_height]))/MAX_PIXEL_VALUE);
      PixelSetGreen(tmp_pixel, ((double)((unsigned char)(*img1)->arrayG[ind_width + (*img1)->width * ind_height]))/MAX_PIXEL_VALUE);
      PixelSetBlue(tmp_pixel,  ((double)((unsigned char)(*img1)->arrayB[ind_width + (*img1)->width * ind_height]))/MAX_PIXEL_VALUE);
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
								
    printf("freeMem start\n");					// there are bugs if free MagikWand memory objects with
    tmp_pixel = DestroyPixelWand(tmp_pixel);			// more 1024x768 resolution
    iterator = DestroyPixelIterator(iterator);	
    mw = DestroyMagickWand(mw);					// free memory magick_wand object
    printf("freeMem end\n");
  
  return 0;
}

int save_cuda_crop_bmp(char *image_full_name, wind_image **img0, wind_image **img1, unsigned char *img)
{	
	resolution height = (*img0)->height + (*img1)->height - 2;
	resolution width = (*img0)->width;
	resolution FileSize = 54 + 3 * width * height; 
	resolution tmp_height = (*img0)->height - 2;
	strcat(image_full_name, BMP_EXTENSION);
	memset(img, 0, sizeof(img));
	resolution i, j, offset0, offset1;
	for(i = 0; i < width; i++)
	for(j = 0; j <= tmp_height; j++)
	{
		offset0 = (i + (height - 1 - j) * width) * 3;
		offset1 = i + width * j;
		img[offset0++] = (unsigned char)((*img0)->arrayB[offset1]);
		img[offset0++] = (unsigned char)((*img0)->arrayG[offset1]);
		img[offset0]   = (unsigned char)((*img0)->arrayR[offset1]);
	}
	for(i = 0; i < width; i++)
	for(j = tmp_height+1; j < height; j++)
	{
		offset0 = (i + (height - 1 - j) * width) * 3;
		offset1 = i + width * (j-tmp_height);
		img[offset0++] = (unsigned char)((*img1)->arrayB[offset1]);
		img[offset0++] = (unsigned char)((*img1)->arrayG[offset1]);
		img[offset0]   = (unsigned char)((*img1)->arrayR[offset1]);
	}

	unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
	unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
	unsigned char bmppad[3] = {0,0,0};
	bmpfileheader[2] = (unsigned char)(FileSize);
	bmpfileheader[3] = (unsigned char)(FileSize >> 8);
	bmpfileheader[4] = (unsigned char)(FileSize >> 16);
	bmpfileheader[5] = (unsigned char)(FileSize >> 24);
	bmpinfoheader[4] = (unsigned char)(width);
	bmpinfoheader[5] = (unsigned char)(width >> 8);
	bmpinfoheader[6] = (unsigned char)(width >> 16);
	bmpinfoheader[7] = (unsigned char)(width >> 24);
	bmpinfoheader[8] = (unsigned char)(height);
	bmpinfoheader[9] = (unsigned char)(height >> 8);
	bmpinfoheader[10] = (unsigned char)(height >>16);
	bmpinfoheader[11] = (unsigned char)(height >>24);
	FILE *file;
	if((file = fopen(image_full_name, "wb")) == NULL)		// open image file descriptor
	{
	  fprintf(stderr, "error open file %s", image_full_name);
	  perror(" ");
	  return -1;
	}
	fwrite(bmpfileheader, 1, 14, file);
	fwrite(bmpinfoheader, 1, 40, file);
	for(i = 0; i < height; i++)
	{
		fwrite(img + (width * i * 3), 3, width, file);
		fwrite(bmppad, 1, (4 - (width * 3) % 4) % 4, file);
	}
	fclose(file);
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