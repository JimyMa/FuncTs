cmake_minimum_required(VERSION 3.22.0 FATAL_ERROR)

# PYTORCH 2.1.0
set(CMAKE_CXX_STANDARD 17)

project(functs)

cmake_policy(SET CMP0074 NEW)

option(ENABLE_FAIT "enable fait backend" OFF)
option(ENABLE_TORCH_TRT "enable torch-trt" OFF)

# Packages
find_package(Torch REQUIRED)
message(${TORCH_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
find_package(CUDAToolkit REQUIRED)

# python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

include_directories(${PROJECT_SOURCE_DIR})

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(CUDAToolkit REQUIRED)
if(ENABLE_TORCH_TRT)
  find_package(torchtrt REQUIRED)
endif(ENABLE_TORCH_TRT)
include_directories(fait)
add_subdirectory(fait/json)

file(GLOB SOURCES
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/ir/*.cpp
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/passes/*.cpp
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/python/*.cpp
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/runtime/*.cpp
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/tensorexpr/*.cpp
  ${PROJECT_SOURCE_DIR}/functs/csrc/*.cpp)

set(FUNCTS_LINK_LIBS
  ${TORCH_LIBRARIES}
)


file(GLOB FAIT_SOURCES
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/passes/fait/*.cpp
  ${PROJECT_SOURCE_DIR}/fait/fuser/*.cpp
  ${PROJECT_SOURCE_DIR}/fait/passes/*.cpp
  ${PROJECT_SOURCE_DIR}/fait/tensorexpr/*.cpp
  ${PROJECT_SOURCE_DIR}/fait/util/*.cpp)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
set(SOURCES ${SOURCES} ${FAIT_SOURCES})

file(GLOB CUPTI_EXT_SOURCES
  ${PROJECT_SOURCE_DIR}/util/cupti_ext/*.cpp
)

file(GLOB CUPTI_EXT_SOURCES
  ${PROJECT_SOURCE_DIR}/fait/util/cupti_ext/*.cpp
)

add_library(cupti_ext SHARED ${CUPTI_EXT_SOURCES})
target_link_libraries(cupti_ext CUDA::cupti CUDA::nvperf_host CUDA::nvperf_target)

add_library(ltprof SHARED ${PROJECT_SOURCE_DIR}/fait/util/profile.cpp ${PROJECT_SOURCE_DIR}/fait/util/metrics.cpp ${CUPTI_EXT_SOURCES})
target_link_libraries(ltprof ${TORCH_LIBRARIES} cupti_ext)

set(FUNCTS_LINK_LIBS 
  ${FUNCTS_LINK_LIBS}
  TorchVision::TorchVision
  nlohmann_json::nlohmann_json
  cupti_ext
)
  

add_library(functs_python SHARED ${SOURCES})
target_link_libraries(functs_python ${FUNCTS_LINK_LIBS})




install(TARGETS functs_python
  DESTINATION lib
)
