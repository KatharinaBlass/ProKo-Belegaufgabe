cmake_minimum_required(VERSION 3.0.0)
project(Belegaufgabe)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} ./src/main.cpp)
target_link_libraries( ${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
