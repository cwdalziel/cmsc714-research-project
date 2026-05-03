# CMSC 714 Research Project

## simgrid install: 
Use the latest version of SimGrid on Linux
Source:
https://github.com/simgrid/simgrid/releases/download/v4.1/simgrid-4.1.tar.gz

Install:
https://simgrid.org/doc/latest/Installing_SimGrid.html#installing-from-source



## simgrid docs:

https://simgrid.org/doc/latest/index.html

mpi specifically: https://simgrid.org/doc/latest/Tutorial_MPI_Applications.html

## simgrid usage

compile with `smpicxx` or `smpicc` and run with `smpirun` which takes the same args as `mpirun` but with `-platform example.xml` to use network topology defined in xml file

## run_benchmarks example usage (after running make)
64 = number of nodes

`
./run_benchmarks.sh bin/alltoall 64
`