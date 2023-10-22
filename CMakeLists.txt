cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# PYTORCH 2.1.0
set(CMAKE_CXX_STANDARD 17)

project(functs)

cmake_policy(SET CMP0074 NEW)

# Packages
find_package(Torch REQUIRED)
message(${TORCH_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
find_package(CUDAToolkit REQUIRED)

# python
find_package(PythonLibs 3.6 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

include_directories(${PROJECT_SOURCE_DIR})

file(GLOB SOURCES
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/ir/*.cpp
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/passes/*.cpp
  ${PROJECT_SOURCE_DIR}/functs/csrc/jit/python/*.cpp
  ${PROJECT_SOURCE_DIR}/functs/csrc/*.cpp)

set(FUNCTS_LINK_LIBS
  ${TORCH_LIBRARIES}
)

add_library(functs_python SHARED ${SOURCES})
target_link_libraries(functs_python ${FUNCTS_LINK_LIBS})

install(TARGETS functs_python
  DESTINATION lib
)
