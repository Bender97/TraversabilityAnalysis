cmake_minimum_required(VERSION 3.1)

project(trav_analysis_2)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

set(CMAKE_FIND_DEBUG_MODE 1)
set(COMPILE_MODE "compile_basic")



# Find installed Open3D, which exports Open3D::Open3D
find_package(OpenCV REQUIRED )
find_package(Boost REQUIRED COMPONENTS thread program_options filesystem)

# Find Eigen3
find_package(Eigen3 REQUIRED)
find_package(Open3D REQUIRED)

add_subdirectory(yaml-cpp)


# include directories
include_directories(
	include
	#	src/utils
	#${catkin_INCLUDE_DIRS}

	${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_LIBRARIES}
    
  yaml-cpp/include

)

# link directories
link_directories(
		include
  ${OpenCV_LIBRARY_DIRS}
  ${Boost_LIBRARIES_DIR}
    yaml-cpp/include
)

ADD_LIBRARY(color_util
	include/ColorUtil.h src/ColorUtil.cpp
)

ADD_LIBRARY(synchro
	include/Synchro.h src/Synchro.cpp
)
target_link_libraries(synchro Open3D::Open3D)

ADD_LIBRARY(metric
	include/Metric.h src/Metric.cpp
)

ADD_LIBRARY(results
	include/Results.h src/Results.cpp
)
target_link_libraries(results metric)


ADD_LIBRARY(feature
	include/Feature.h src/Feature.cpp
)

ADD_LIBRARY(normalizer
	include/Normalizer.h src/Normalizer.cpp
)
target_link_libraries(normalizer yaml-cpp ${OpenCV_LIBRARIES})


ADD_LIBRARY(traindataset
	include/TrainDataset.h src/TrainDataset.cpp
)
target_link_libraries(traindataset feature normalizer metric yaml-cpp ${OpenCV_LIBRARIES})

ADD_LIBRARY(cylinder
	include/Cylinder.h src/Cylinder.cpp
)
target_link_libraries(cylinder normalizer feature yaml-cpp traindataset Open3D::Open3D  synchro)

SET (custom_libs cylinder feature normalizer color_util yaml-cpp metric results z)



# Produce Feature
add_executable(mtmain src/mtmain.cpp)
target_link_libraries(mtmain ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} ${custom_libs} ${OpenCV_LIBRARIES} )
#
# Train
add_executable(train src/train.cpp)
target_link_libraries(train ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} ${custom_libs} ${OpenCV_LIBRARIES} )
# Test
add_executable(test src/try_cyl.cpp)
target_link_libraries(test ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} ${custom_libs} ${OpenCV_LIBRARIES}  Open3D::Open3D )

# add_executable(test_dl src/test_labels.cpp)
# target_link_libraries(test_dl ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} ${custom_libs} ${OpenCV_LIBRARIES}  Open3D::Open3D )