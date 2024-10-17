#############################################################
#                                                           #
# Copyright (c) 2022, University of California, Los Angeles #
#                                                           #
# Authors: Kenny J. Chen, Brett T. Lopez                    #
# Contact: kennyjchen@ucla.edu, btlopez@ucla.edu            #
#                                                           #
#############################################################

cmake_minimum_required(VERSION 3.10.0)
project(trlo)
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

#add_compile_options(-std=c++14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O3 -g -pthread")
#set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -g -pthread")

# PCL
find_package( PCL REQUIRED )
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
link_libraries(${CERES_LIBRARIES})

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_libraries(${OpenCV_LIBRARIES})

# Eigen3
find_package( Eigen3 REQUIRED )
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenMP
include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
  message("ERROR: OpenMP could not be found.")
endif(OPENMP_FOUND)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# CUDA
find_package(CUDA REQUIRED)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.3)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)

# TENSORRT install path
set(TENSORRT_ROOT /your/TensorRT-8.5.3.1/path)
set(TENSORRT_INCLUDE_DIRS ${TENSORRT_ROOT}/include/)
set(TENSORRT_LIBRARY_DIRS ${TENSORRT_ROOT}/lib)

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    3rdparty/libspconv/include
    include
)

link_directories(
	${TENSORRT_LIBRARY_DIRS}
  3rdparty/libspconv/lib/${arch}
	/usr/lib/aarch64-linux-gnu
  /usr/aarch64-linux-gnu/lib/
)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  pcl_ros
  message_generation
  jsk_recognition_msgs
)


add_service_files(
  DIRECTORY srv
  FILES
  save_pcd.srv
  save_traj.srv
)

generate_messages()

catkin_package(
  CATKIN_DEPENDS
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    pcl_ros
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
    nano_gicp
    nanoflann
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_LIBRARIES})

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
file(WRITE ${CMAKE_BINARY_DIR}/test_cpuid.cpp "#include <cpuid.h>")
try_compile(HAS_CPUID ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
file(REMOVE ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
if(HAS_CPUID)
  add_compile_definitions(HAS_CPUID)
endif()

file(GLOB_RECURSE CENTER_POINTPILLARS_FILES
    src/center_pointpillars/*.cu
    src/center_pointpillars/*.cpp
)

# 3D MOT
# set(3D_MOT_FILES src/3d_mot/3d_mot_node.cpp src/3d_mot/imm_ukf_jpda.cpp src/3d_mot/ukf.cpp )
# add_executable(3d_mot_node ${3D_MOT_FILES})
# target_link_libraries(3d_mot_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

# Center_PointPillarss Node
set(3D_MOT_FILES src/3d_mot/imm_ukf_jpda.cpp src/3d_mot/ukf.cpp )
cuda_add_executable(centerpp_node src/centerpp_node/centerpp_node.cpp ${CENTER_POINTPILLARS_FILES} ${3D_MOT_FILES})
target_link_libraries(centerpp_node
    libnvinfer.so
    libnvonnxparser.so
    libspconv.so
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
)

# NanoFLANN
add_library(nanoflann STATIC
  src/nano_gicp/nanoflann.cc
)
target_link_libraries(nanoflann ${PCL_LIBRARIES})
target_include_directories(nanoflann PUBLIC include ${PCL_INCLUDE_DIRS})

# NanoGICP
add_library(nano_gicp STATIC
  src/nano_gicp/lsq_registration.cc
  src/nano_gicp/nano_gicp.cc
)
#target_link_libraries(nano_gicp ${PCL_LIBRARIES} ${OpenMP_LIBS} nanoflann)
target_link_libraries(nano_gicp ${PCL_LIBRARIES} OpenMP::OpenMP_CXX nanoflann)
target_include_directories(nano_gicp PUBLIC include ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

# Odometry Node
add_executable(trlo_odom_node src/trlo/odom_node.cc src/trlo/odom.cc)
add_dependencies(trlo_odom_node ${catkin_EXPORTED_TARGETS})
target_compile_options(trlo_odom_node PRIVATE ${OpenMP_FLAGS})
target_link_libraries(trlo_odom_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads nano_gicp)

# Mapping Node
add_executable (trlo_map_node src/trlo/map_node.cc src/trlo/map.cc)
add_dependencies(trlo_map_node ${catkin_EXPORTED_TARGETS})
target_compile_options(trlo_map_node PRIVATE ${OpenMP_FLAGS})
target_link_libraries(trlo_map_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads)
