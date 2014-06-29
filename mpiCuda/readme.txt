Task: 
there are images at shared network folder, that need to be filtered by 
many PC with optimal end time. Ideally when all PCs end filtering work 
at the same time. Each PC filters image only on cuda videocard. 
Each PC can processing only integer number (count) of images - one image cannot 
be processed by many PCs, only by one PC. Program must use MPI. 
You must define IPv4 addresses for all PCs at network + images shared folder. 
Algorithm: 
- one PC (main PC) from local network runs the program 
- this main PC get resolution from each image in images shared folder 
- then associates each image to the specific PC for processing according count 
of PCs and each image resolution 
- main PC process creates 2 specific files: hostfile and imagelistfile 
<hostfile> - contains IPv4 addresses of each PC in local network and count of 
process that will be created using MPI, one process processing one image 
<imagelistfile> - contains imageFileNames in specific MPI order, 
that each PC filters it's own images only 
- main PC run 'mpirun' process with created early <hostfile> 
- each PC compute his own images
- each PC send MPI message to main PC that his own work is terminated 
(that all his own PC's images is filtered) 
- main PC process will e terminated, when main PC get terminated messages 
from all PCs in local network
