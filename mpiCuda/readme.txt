[COMPILE]:
mpicc -c mpiFilter.c -o mpiFilter.o `pkg-config --cflags --libs MagickWand`
nvcc -c cuda.cu -o cuda.o -arch sm_20
mpicc mpiFilter.o cuda.o -o mpiFilter `pkg-config --cflags --libs MagickWand` -L/opt/cuda-6.0/lib64 -lcudart
gcc allocate.c -o startMpiFilter `pkg-config --cflags --libs MagickWand`

[RUN]:
./startMpiFilter </full_path/images_path> [hosts_ipv4] <mode>
<mode>:
-cpu[s] == cpu filter
-shared[s] == cuda shared memory filter
-async[s] == cuda async filter
[s] - if last symbol is 's', then save result to file

/*
 * WARNING! result images folder ('<images_path>/result') must be created manually
 * if it's not, then you will get error at execution program
 */
 
 
example_run:
./startMpiFilter /mnt/studpublic/images 192.168.1.10 192.168.1.9 -asyncs

/*
 * WARNING! you need to create result images folder '/mnt/studpublic/images/result' manually
 * if it's not, then you will get error at execution program example
 */


