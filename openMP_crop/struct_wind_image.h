#include <stdio.h>
typedef signed int pixel_channel;
typedef unsigned long resolution;

typedef struct
{
  resolution height;
  resolution width;
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


void fake_new_wind_image(wind_image **img, unsigned long image_width,  unsigned long image_height)
{
  free_image(img);
  (*img) = (wind_image*)malloc(sizeof(wind_image) * 1);
  (*img)->width = image_width;
  (*img)->height = image_height;
  (*img)->arrayR = (pixel_channel*)malloc((*img)->width * (*img)->height * sizeof(pixel_channel));
  (*img)->arrayG = (pixel_channel*)malloc((*img)->width * (*img)->height * sizeof(pixel_channel));
  (*img)->arrayB = (pixel_channel*)malloc((*img)->width * (*img)->height * sizeof(pixel_channel));
}
