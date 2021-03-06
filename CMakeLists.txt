cmake_minimum_required(VERSION 3.0.2)
project(object_detection_openvino)

# Boost
find_package(Boost REQUIRED)

# OpenCV
find_package(OpenCV 3 REQUIRED COMPONENTS core imgproc imgcodecs highgui QUIET)

# Ngraph and InferenceEngine
find_package(ngraph REQUIRED)
find_package(InferenceEngine 2.0 REQUIRED)
if(NOT(InferenceEngine_FOUND))
    message(WARNING "InferenceEngine is disabled or not found.")
    return()
else()
    include_directories(${InferenceEngine_INCLUDE_DIRS}/../samples/cpp/common/)
    #include_directories(${InferenceEngine_INCLUDE_DIRS}/../samples/cpp/common/utils/include/)
endif()

if(TARGET IE::ie_cpu_extension)
    add_definitions(-DWITH_EXTENSIONS)
endif()

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
    roscpp
    nodelet
    std_msgs
    sensor_msgs
    geometry_msgs
    cv_bridge
    image_transport
    message_filters
    vision_msgs
    pcl_ros
    pcl_conversions
)


################################################
## Declare ROS messages, services and actions ##
################################################

###################################
## catkin specific configuration ##
###################################
# The catkin_package macro generates cmake config files for your package
# Declare things to be passed to dependent projects
catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS
        roscpp
        nodelet
        std_msgs
        sensor_msgs
        geometry_msgs
        cv_bridge
        image_transport
        message_filters
        vision_msgs
        pcl_ros
        pcl_conversions
    DEPENDS Boost
)

###########
## Build ##
###########

# Specify additional locations of header files
include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
)

# Set SOURCES
set(SOURCES
    src/objectDetectionVPU.cpp
    src/openvino.cpp
)

# Declare a C++ executable 
add_executable(${PROJECT_NAME}_node ${SOURCES} src/objectDetectionVPU_node.cpp)

# Declare a C++ library
add_library(${PROJECT_NAME}_nodelet ${SOURCES} src/objectDetectionVPU_nodelet.cpp)

# Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node
    PRIVATE 
    ${InferenceEngine_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${NGRAPH_LIBRARIES}
    ${Boost_LIBRARIES}
    ${catkin_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME}_nodelet
    PRIVATE 
    ${InferenceEngine_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${NGRAPH_LIBRARIES}
    ${Boost_LIBRARIES}
    ${catkin_LIBRARIES}
)

if(TARGET IE::ie_cpu_extension)
    target_link_libraries(${PROJECT_NAME} IE::ie_cpu_extension)
endif()


#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}_node ${PROJECT_NAME}_nodelet
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

## Mark launch files for installation
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
