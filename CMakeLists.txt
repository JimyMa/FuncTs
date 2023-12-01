cmake_minimum_required(VERSION 3.22.0 FATAL_ERROR)

# PYTORCH 2.1.0
set(CMAKE_CXX_STANDARD 17)

project(functs)

cmake_policy(SET CMP0074 NEW)

option(ENABLE_FUNCTS_PYTHON "enable functs python wrapper" OFF)
option(ENABLE_NNC_INT_LIST "need to revise pytorch code to support List[int] in NNC" OFF)

# Packages
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

include_directories(${PROJECT_SOURCE_DIR})

find_package(Torch REQUIRED)

# find_package(TorchVision REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_library(CUDA_NVPERF_HOST nvperf_host PATHS ${CUDAToolkit_LIBRARY_DIR} REQUIRED)
find_library(CUDA_NVPERF_TARGET nvperf_target PATHS ${CUDAToolkit_LIBRARY_DIR} REQUIRED)

include_directories(fait)
add_subdirectory(fait/json)

file(GLOB SOURCES
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/api/*.cpp
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/ir/*.cpp
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/passes/*.cpp
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/runtime/*.cpp
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/tensorexpr/*.cpp)

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
  ${PROJECT_SOURCE_DIR}/fait/util/cupti_ext/*.cpp
)

file(GLOB PROFILER_SOURCES
  ${PROJECT_SOURCE_DIR}/fait/util/profile.cpp
  ${PROJECT_SOURCE_DIR}/fait/util/metrics.cpp
)

add_library(cupti_ext SHARED ${CUPTI_EXT_SOURCES})
target_link_libraries(cupti_ext CUDA::cupti ${CUDA_NVPERF_HOST} ${CUDA_NVPERF_TARGET})

add_library(ltprof SHARED ${PROFILER_SOURCES} ${CUPTI_EXT_SOURCES})
target_link_libraries(ltprof ${TORCH_LIBRARIES} cupti_ext)

set(FUNCTS_LINK_LIBS
  ${TORCH_LIBRARIES}
  nlohmann_json::nlohmann_json
  cupti_ext
)

add_library(functs SHARED ${SOURCES})
target_link_libraries(functs ${FUNCTS_LINK_LIBS})

if(ENABLE_NNC_INT_LIST)
target_compile_definitions(functs PRIVATE ENABLE_NNC_INT_LIST)
endif(ENABLE_NNC_INT_LIST)

install(TARGETS functs
  DESTINATION lib
)

install(TARGETS cupti_ext
  DESTINATION lib
)

install(TARGETS ltprof
  DESTINATION lib
)

if(ENABLE_FUNCTS_PYTHON)
  include_directories(${PYTHON_INCLUDE_DIR})
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
  file(GLOB PY_SOURCES
    ${PROJECT_SOURCE_DIR}/functs/csrc/jit/python/*.cpp
    ${PROJECT_SOURCE_DIR}/functs/csrc/*.cpp
  )
  add_library(functs_python SHARED ${PY_SOURCES})
  target_link_libraries(functs_python functs ${PYTHON_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
  install(TARGETS functs_python
    DESTINATION lib
  )
endif(ENABLE_FUNCTS_PYTHON)
