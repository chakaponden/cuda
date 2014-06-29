Task: 
image filter using cpu and gpu compute at the same time + 
need: cpu filter processing time == gpu filter processing time at one image. 
Cpu filter must use openMP. 
For this, program crop image to 2 halfs: 
upper half - for cpu processing 
lower half - for gpu processing 
Image grops only by line of pixel (pixel[image_width]). 
To determine in what proportions crop image to 2 halfs 
use ratio == multiplier == gpu_time_filter_processing/cpu_time_filter_processing. 
But this multiplier is not the same for all image resolutons, then 
program calculate multipliers for many image resolution (you define resolutions at initBD run arguments) 
by initBD program that creates database of multipliers (initBD.txt and initBD.bin files). 
This files (initBD.bin for openMP program and initBD.txt for human view) use to 
crop image to gpu_half and cpu_half for processing. 
Gpu and cpu processing execute at separated threads, main thread waits for their terminate.
Algorhutm for all images at defined folder: 
- image crop to 2 halfs - cpu && gpu processing halfs (main thread)
- cpu and gpu filter his own half (create new 2 threads: first for cpu filter, second for gpu filter)
- resulted halfs of images attach together to one result image (main thread)
- save one result image to file (main thread)