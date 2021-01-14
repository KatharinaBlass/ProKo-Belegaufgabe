#include <iostream>
#include <string.h>
#include <stdio.h>
#include "./VersionMPI.hpp"
#include "./VersionOpenCV.hpp"
#include "./VersionOpenMP.hpp"
#include "./VersionNonParallel.hpp"

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " OPTION (f.e. mpi, omp, mpiOmp, cv, npSlow, npFast) "
              << " IMAGE_PATH" << std::endl;
    return 1;
  }

  if (strcmp(argv[1], "npSlow") == 0)
  {
    VersionNonParallel(argc, argv, true);
  }
  else if (strcmp(argv[1], "npFast") == 0)
  {
    VersionNonParallel(argc, argv, false);
  }
  else if (strcmp(argv[1], "omp") == 0)
  {
    VersionOpenMP(argc, argv);
  }
  else if (strcmp(argv[1], "mpi") == 0)
  {
    VersionMPI(argc, argv, false);
  }
  else if (strcmp(argv[1], "mpiOmp") == 0)
  {
    VersionMPI(argc, argv, true);
  }
  else if (strcmp(argv[1], "cv") == 0)
  {
    VersionOpenCV(argc, argv);
  }

  return 0;
}