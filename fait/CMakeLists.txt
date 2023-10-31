cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# PYTORCH 2.1.0
set(CMAKE_CXX_STANDARD 17)

# # PYTORCH 2.0.0
# set(CMAKE_CXX_STANDARD 14)

project(long-tail)

cmake_policy(SET CMP0074 NEW)

# Options
option(ENABLE_TORCH_TRT "enable torch-trt" OFF)

# Packages
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(CUDAToolkit REQUIRED)
if(ENABLE_TORCH_TRT)
  find_package(torchtrt REQUIRED)
endif(ENABLE_TORCH_TRT)

# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(json)

include_directories(${PROJECT_SOURCE_DIR})

file(GLOB SOURCES
  ${PROJECT_SOURCE_DIR}/fuser/*.cpp
  ${PROJECT_SOURCE_DIR}/passes/*.cpp
  ${PROJECT_SOURCE_DIR}/tensorexpr/*.cpp
  ${PROJECT_SOURCE_DIR}/util/*.cpp)

set(LONG_TAIL_LINK_LIBS 
  ${TORCH_LIBRARIES}
  TorchVision::TorchVision
  nlohmann_json::nlohmann_json
)

if(ENABLE_TORCH_TRT)
  set(LONG_TAIL_LINK_LIBS ${LONG_TAIL_LINK_LIBS} torchtrt)
endif(ENABLE_TORCH_TRT)

file(GLOB CUPTI_EXT_SOURCES
  ${PROJECT_SOURCE_DIR}/util/cupti_ext/*.cpp
)

add_library(cupti_ext SHARED ${CUPTI_EXT_SOURCES})
target_link_libraries(cupti_ext CUDA::cupti CUDA::nvperf_host CUDA::nvperf_target)

set(LONG_TAIL_LIB long-tail)
add_library(${LONG_TAIL_LIB} SHARED ${SOURCES})
target_link_libraries(${LONG_TAIL_LIB} ${LONG_TAIL_LINK_LIBS} cupti_ext)

add_library(ltprof SHARED ${PROJECT_SOURCE_DIR}/util/profile.cpp ${PROJECT_SOURCE_DIR}/util/metrics.cpp ${CUPTI_EXT_SOURCES})
target_link_libraries(ltprof ${TORCH_LIBRARIES} cupti_ext)

# add_executable(fait ${PROJECT_SOURCE_DIR}/fait.cpp)
# target_link_libraries(fait long-tail)

# add_executable(run_ts ${PROJECT_SOURCE_DIR}/run_ts.cpp)
# target_link_libraries(run_ts long-tail)

# add_executable(test_lowering ${PROJECT_SOURCE_DIR}/testing/test_lowering.cpp)
# target_link_libraries(test_lowering long-tail)

# if(ENABLE_TORCH_TRT)
#   add_executable(run_trt ${PROJECT_SOURCE_DIR}/run_trt.cpp)
#   target_link_libraries(run_trt long-tail)

#   add_executable(run_net_trt ${PROJECT_SOURCE_DIR}/run_net_trt.cpp)
#   target_link_libraries(run_net_trt long-tail)
# endif(ENABLE_TORCH_TRT)
