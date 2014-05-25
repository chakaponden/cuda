[COMPILE]:
mpicc -c mpiFilter.c -o mpiFilter.o `Wand-config --cflags`
nvcc -c cuda.cu -o cuda.o -arch sm_50
mpicc mpiFilter.o cuda.o -o mpiFilter `Wand-config --cflags --libs` -L/opt/cuda-6.0/lib64 -lcudart
gcc allocate.c -o startMpiFilter `MagickWand-config --cflags --ldflags`

[RUN]:
./startMpiFilter <image_path> [hosts_ipv4] <mode>
<mode>:
-cpu[s] == cpu filter
-shared[s] == cuda shared memory filter
-async[s] == cuda async filter
[s] - if last symbol is 's', then save result to file

example_run:
./startMpiFilter $HOME/images 192.168.1.10 192.168.1.9 -asyncs
