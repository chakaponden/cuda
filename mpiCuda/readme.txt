[COMPILE]:
cc `MagickWand-config --cflags --cppflags` -o startMpiFilter allocate.c `MagickWand-config --ldflags --libs`
mpicc -o mpiFilter mpiFilter.c -std=gnu99 -lm

[RUN]:
./startMpiFilter <image_path> [hosts_ipv4]

example_run:
./startMpiFilter $HOME/images 192.168.1.10 192.168.1.9
