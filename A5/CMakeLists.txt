cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(template LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

# check builder directory
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Please select another folder to build the project.")
endif()


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message("Setting build type to 'RelWithDebInfo' as none was specified.")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
    # force configuration set in VS
    set(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE} CACHE STRING "" FORCE)
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")


set(cuda_runtime_DIR $ENV{CUDA_PATH}/lib64)
message(STATUS "Runtime DIR: ${cuda_runtime_DIR}")

# find CUDA library
find_package(CUDA REQUIRED)
if(${CUDA_FOUND})
    # set(CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=sm_52;-O2;-G;-g")#根据具体GPU性能更改算力参数
    # Only when you have CUDA 8 or newer version (--expt-extended-lambda or --extended-lambda)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-extended-lambda;-Xcompiler;-fPIC")
else(${CUDA_FOUND})
    message(STATUS "cuda not found!")
endif()

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)


# remove last end of "/"
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
# get above dir name
get_filename_component(SECOND_FOLDER_ABSOLUTE ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)
# common commponent
set(COMMON_DIR_HEADERS 
    ${SECOND_FOLDER_ABSOLUTE}/common 
    ${SECOND_FOLDER_ABSOLUTE}/common/GL
    )

file(GLOB CURRENT_DIR_SOURCES src/*.cpp src/*.cu)

# Generate executable file
cuda_add_executable(template ${CURRENT_DIR_SOURCES})

target_include_directories(template 
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${COMMON_DIR_HEADERS}
        ${CUDA_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        ${GLUT_INCLUDE_DIRS}
    )

target_link_directories(template PRIVATE $ENV{CUDA_PATH}/targets/x86_64-linux/lib)
target_link_libraries(template ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})



set_target_properties(template PROPERTIES
    OUTPUT_NAME "heatTransfer3D"
    CUDA_SEPARABLE_COMPILATION ON
    )