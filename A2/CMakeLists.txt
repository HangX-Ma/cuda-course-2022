cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(template LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

if(${CMAKE_CURRENT_BINARY_DIR})
    
endif()

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

# find CUDA library
find_package(CUDA REQUIRED)
if(${CUDA_FOUND})
    # set(CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS -arch=sm_61;-O1;-G;-g;-std=c++11)#根据具体GPU性能更改算力参数
    #SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;-O3;-G;-g)
else(${CUDA_FOUND})
    message(STATUS "cuda not found!")
endif()

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)


file(GLOB_RECURSE CURRENT_DIR_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_DIR_SOURCES  *.cpp *.cu)


# remove last end of "/"
string(REGEX REPLACE "/$" "" CURRENT_FOLDER_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR})
# get above dir name
get_filename_component(SECOND_FOLDER_ABSOLUTE ${CURRENT_FOLDER_ABSOLUTE} DIRECTORY)
# common commponent
set(COMMON_DIR_HEADERS 
    ${SECOND_FOLDER_ABSOLUTE}/common 
    ${SECOND_FOLDER_ABSOLUTE}/common/GL
    )


# Generate executable file
cuda_add_executable(template ${CURRENT_DIR_SOURCES} ${CURRENT_DIR_HEADERS} ${COMMON_DIR_HEADERS})

target_include_directories(template 
    PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${COMMON_DIR_HEADERS}
        ${OPENGL_INCLUDE_DIRS}
        ${GLUT_INCLUDE_DIRS}
    )

target_link_libraries(template ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

target_link_directories(template 
    PRIVATE 
        $ENV{CUDA_PATH}/lib64
    )

set_target_properties(template PROPERTIES
    OUTPUT_NAME "dynamic_ray_tracer")