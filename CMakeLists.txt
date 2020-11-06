cmake_minimum_required(VERSION 3.0.0)
project(Belegaufgabe)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})

file(GLOB SOURCES "src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries( ${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX ${OpenCV_LIBS} ${MPI_LIBRARIES} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)