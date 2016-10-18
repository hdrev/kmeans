cmake_minimum_required(VERSION 3.6)
project(kmeans)

set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -Wall -lm -std=c99")

set(SOURCE_FILES kmeans.c)
add_executable(kmeans ${SOURCE_FILES})
IF(UNIX)
    target_link_libraries(kmeans m)
ENDIF(UNIX)
