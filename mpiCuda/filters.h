#include <time.h>
#include <wand/magick_wand.h>

#define KERNEL_SIZE 3
#define MAX_PIXEL_VALUE 255

typedef signed int pixel_channel;

int kernel[KERNEL_SIZE][KERNEL_SIZE] =				{  0, -1,  0,
								  -1,  5, -1, 
								   0, -1,  0 };

#define ThrowWandException(wand) \
{ \
        char *description; \
        ExceptionType severity; \
        description=MagickGetException(wand,&severity); \
        printf("\n\n-----\n%s %s %lu %s\n",GetMagickModule(),description); \
        description=(char *) MagickRelinquishMemory(description); \
        exit(-1);\
}

typedef struct
{
  unsigned long height;
  unsigned long width;
  pixel_channel *arrayR;
  pixel_channel *arrayG;
  pixel_channel *arrayB;  
} wind_image;

void free_image(wind_image **img)
{
  if((*img) != NULL)
  {
    if((*img)->height > 0 && (*img)->width > 0)
    {
      free((*img)->arrayR);
      free((*img)->arrayG);
      free((*img)->arrayB);
      (*img)->height = 0;
      (*img)->width = 0; 
    }
    free((*img));
    (*img) = NULL;
  }
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
  unsigned long ind_height, ind_width, tmp_width;
  PixelIterator *iterator = NULL;
  if((iterator = NewPixelIterator(mw)) == NULL)
  {   
    mw = DestroyMagickWand(mw);					// free memory magick_wand object
    fclose(f);
    ThrowWandException(mw);    
  }
  PixelWand **pixels = NULL;
  for(ind_height = 0; ind_height < (*img)->height; ind_height++)	// get RGB channels for each pixel
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
  fclose(f);							// close file descriptor  
  return 0;
}

int save_image(char *image_full_name, wind_image **img)
{
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
  unsigned long ind_height, ind_width, tmp_width;  
  for(ind_height = 0; ind_height < (*img)->height; ind_height++)	// get RGB channels for each pixel
  {
    pixels = PixelGetNextIteratorRow(iterator, &tmp_width);	// get row pixels
    for(ind_width = 0; ind_width < (*img)->width; ind_width++)	// get RGB channels each pixel in row
    {
      PixelSetRed(tmp_pixel, (double)(((double)(*img)->arrayR[ind_width + (*img)->width * ind_height])/MAX_PIXEL_VALUE));
      PixelSetGreen(tmp_pixel, (double)(((double)(*img)->arrayG[ind_width + (*img)->width * ind_height])/MAX_PIXEL_VALUE));
      PixelSetBlue(tmp_pixel, (double)(((double)(*img)->arrayB[ind_width + (*img)->width * ind_height])/MAX_PIXEL_VALUE));
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
  //iterator = DestroyPixelIterator(iterator);
  tmp_pixel = DestroyPixelWand(tmp_pixel);
  mw = DestroyMagickWand(mw);					// free memory magick_wand object
  return 0;
}

double cpu_filter(wind_image **src_img, wind_image **result_img)
{
  unsigned long width = (*src_img)->width;
  unsigned long height = (*src_img)->height;
  unsigned long x, y, i, j, pixelPosX, pixelPosY;
  pixel_channel r, g, b, rSum, gSum, bSum;
  clock_t begin, end;
  free_image(result_img);
  (*result_img) = (wind_image*)malloc(sizeof(wind_image));
  (*result_img)->height = height;
  (*result_img)->width = width;
  (*result_img)->arrayR = (pixel_channel*)malloc(width * height * sizeof(pixel_channel));
  (*result_img)->arrayG = (pixel_channel*)malloc(width * height * sizeof(pixel_channel));
  (*result_img)->arrayB = (pixel_channel*)malloc(width * height * sizeof(pixel_channel));
  
  begin = clock();
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
  return ((double)(end - begin) / CLOCKS_PER_SEC);
}

//unsigned int* Shared_Memory_Convolution(unsigned int *channel, unsigned long width, unsigned long height, int kernel[3][3], float *time);


double cuda_shared_memory(wind_image **src_img, wind_image **result_img)
{
  unsigned long width = (*src_img)->width;
  unsigned long height = (*src_img)->height;
  free_image(result_img);
  (*result_img) = (wind_image*)malloc(sizeof(wind_image));
  (*result_img)->height = height;
  (*result_img)->width = width;
  (*result_img)->arrayR = (pixel_channel*)malloc(width * height * sizeof(pixel_channel));
  (*result_img)->arrayG = (pixel_channel*)malloc(width * height * sizeof(pixel_channel));
  (*result_img)->arrayB = (pixel_channel*)malloc(width * height * sizeof(pixel_channel));
  float time = 0, all_time = 0;
  *((*result_img)->arrayR) = Shared_Memory_Convolution(((*src_img)->arrayR), width, height, kernel, &time);
  all_time += time;
  *((*result_img)->arrayG) = Shared_Memory_Convolution(((*src_img)->arrayG), width, height, kernel, &time);
  all_time += time;
  *((*result_img)->arrayB) = Shared_Memory_Convolution(((*src_img)->arrayB), width, height, kernel, &time);
  return (double)all_time;
}
