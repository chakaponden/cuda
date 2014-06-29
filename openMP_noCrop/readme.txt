Task: 
image filter using cpu and gpu compute at the same time + 
need: cpu filter processing time == gpu filter processing time at one image. 
Cpu filter must use openMP. 
At this program cpu filter processing one image from folder and gpu filter processing 
one image from folder, but this images are various - cpu filter his own image, gpu his own.
Images from folder devided into 2 types: 
- image for cpu processing 
- image for gpu processing 
To determine, which image type is, program calculate 
ratio == multiplier == gpu_time_filter_processing/cpu_time_filter_processing. 
Image 'weight' is determined by image resolution (count of pixels). 
And according this multiplier devides images into 2 groups.
Gpu and cpu processing execute at separated processes.
Algorhutm: 
- run filter testImage for calculate multiplier = gpu_time/cpu_time 
- read all image resolution from defined folder 
- devide all images into 2 groups (for cpu processing and for gpu processing) 
- run filter and save result all images in folder, cpu filters his own images, gpu filters his own