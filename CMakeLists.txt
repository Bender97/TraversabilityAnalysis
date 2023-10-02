cmake_minimum_required(VERSION 3.1)

project(trav_analysis_2)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

#set(COMPILE_MODE "compile_basic")

option(OPEN3D "Compile with Open3D flag" 1)
if (OPEN3D)
add_definitions(-DOPEN3D=1)
else()
add_definitions(-DOPEN3D=0)
endif()

find_package(OpenCV REQUIRED )

# Find Eigen3
find_package(Eigen3 REQUIRED)

if(OPEN3D)
	# Find installed Open3D, which exports Open3D::Open3D
	find_package(Open3D REQUIRED)
	SET (open3d_lib Open3D::Open3D)
	message("OPEN3D IS 1")
else()
	SET (open3d_lib "")
	message("OPEN3D IS ZERO")
endif()



find_package(yaml-cpp REQUIRED)

#add_subdirectory(yaml-cpp)


# include directories
include_directories(
	include
	${EIGEN3_INCLUDE_DIR}
	${OpenCV_INCLUDE_DIRS}
	yaml-cpp/include
)

ADD_LIBRARY(color_util
	include/ColorUtil.h src/ColorUtil.cpp
)

ADD_LIBRARY(dataloader
	include/DataLoader.h src/DataLoader.cpp
)
target_link_libraries(dataloader color_util yaml-cpp ${open3d_lib})

if(OPEN3D)
	ADD_LIBRARY(synchro
		include/Synchro.h src/Synchro.cpp
	)
	target_link_libraries(synchro dataloader yaml-cpp ${open3d_lib})
	set(synchro_lib synchro)
else()
	set(synchro_lib "")
endif()


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
target_link_libraries(normalizer yaml-cpp ${OpenCV_LIBS})


ADD_LIBRARY(cylinder
	include/Cylinder.h src/Cylinder.cpp
)
target_link_libraries(cylinder ${OpenCV_LIBS} normalizer feature metric dataloader yaml-cpp)


ADD_LIBRARY(traindataset
	include/TrainDataset.h src/TrainDataset.cpp
)
target_link_libraries(traindataset yaml-cpp ${OpenCV_LIBS} feature normalizer metric cylinder )

SET (custom_libs cylinder feature normalizer color_util yaml-cpp metric results dataloader ${synchro_lib})



# # Produce Feature
add_executable(mtmain src/mtmain.cpp)
target_link_libraries(mtmain ${custom_libs} ${OpenCV_LIBS} )

# Train
add_executable(train src/train.cpp)
target_link_libraries(train ${custom_libs} traindataset ${OpenCV_LIBS} )

# # Test
add_executable(test src/try_cyl.cpp)
target_link_libraries(test  ${custom_libs})

add_executable(test_dl src/test_labels.cpp)
target_link_libraries(test_dl ${custom_libs} )

add_executable(processSingleCloud src/processSingleCloud.cpp)
target_link_libraries(processSingleCloud ${custom_libs} )