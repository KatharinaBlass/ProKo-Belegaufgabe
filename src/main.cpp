#include <iostream>
#include <string.h>
#include <stdio.h>
#include "./mpiVersion.hpp"
#include "./cvVersion.hpp"
#include "./ompVersion.hpp"
#include "./normalVersion.hpp"

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " OPTION (f.e. mpi, omp, mpiOmp, cv, normalSlow, normalFast) "
              << " IMAGE_PATH" << std::endl;
    return 1;
  }

  if (strcmp(argv[1], "normalSlow") == 0)
  {
    normalVersion(argc, argv, true);
  }
  else if (strcmp(argv[1], "normalFast") == 0)
  {
    normalVersion(argc, argv, false);
  }
  else if (strcmp(argv[1], "omp") == 0)
  {
    ompVersion(argc, argv);
  }
  else if (strcmp(argv[1], "mpi") == 0)
  {
    mpiVersion(argc, argv, false);
  }
  else if (strcmp(argv[1], "mpiOmp") == 0)
  {
    mpiVersion(argc, argv, true);
  }
  else if (strcmp(argv[1], "cv") == 0)
  {
    cvVersion(argc, argv);
  }

  return 0;
}