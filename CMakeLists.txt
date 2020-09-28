cmake_minimum_required(VERSION 2.8.12)
project(object_detection_openvino)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

# Boost
find_package(Boost REQUIRED)

# OpenCV
find_package(OpenCV 3 REQUIRED COMPONENTS opencv_core opencv_imgproc opencv_imgcodecs highgui QUIET)
if(NOT(OpenCV_FOUND))
    message(WARNING "OpenCV is disabled or not found.")
    return()
else()
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# InferenceEngine
find_package(InferenceEngine 2.0 QUIET)
if(NOT(InferenceEngine_FOUND))
    message(WARNING "InferenceEngine is disabled or not found.")
    return()
else()
    include_directories(${InferenceEngine_INCLUDE_DIRS})
    include_directories(${InferenceEngine_INCLUDE_DIRS}/../samples/common/)
    include_directories(${InferenceEngine_INCLUDE_DIRS}/../src/extension/)
endif()

if(TARGET IE::ie_cpu_extension)
    add_definitions(-DWITH_EXTENSIONS)
endif()

# Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    std_msgs
    geometry_msgs
    cv_bridge    
    image_transport
    message_generation    
    message_filters
    visualization_msgs
)


################################################
## Declare ROS messages, services and actions ##
################################################

# To declare and build messages, services or actions from within this
# package, follow these steps:
# * Let MSG_DEP_SET be the set of packages whose message types you use in
#   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
# * In the file package.xml:
#   * add a build_depend tag for "message_generation"
#   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
#   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
#     but can be declared for certainty nonetheless:
#     * add a exec_depend tag for "message_runtime"
# * In this file (CMakeLists.txt):
#   * add "message_generation" and every package in MSG_DEP_SET to
#     find_package(catkin REQUIRED COMPONENTS ...)
#   * add "message_runtime" and every package in MSG_DEP_SET to
#     catkin_package(CATKIN_DEPENDS ...)
#   * uncomment the add_*_files sections below as needed
#     and list every .msg/.srv/.action file to be processed
#   * uncomment the generate_messages entry below
#   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

# Generate messages in the 'msg' folder
add_message_files(
	DIRECTORY msg
	FILES
	BoundingBox.msg
	BoundingBoxArray.msg
	BoundingBox3d.msg
	BoundingBox3dArray.msg
)

# Generate added messages and services with any dependencies listed here
generate_messages( 
	DEPENDENCIES
	std_msgs
)


###################################
## catkin specific configuration ##
###################################
# The catkin_package macro generates cmake config files for your package
# Declare things to be passed to dependent projects
catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS        
        roscpp
        roslib
        std_msgs
        geometry_msgs
        cv_bridge
        image_transport
        message_runtime        
        message_filters
        visualization_msgs
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

# Declare a C++ executable
add_executable(${PROJECT_NAME}
  src/objectDetectionOpenvino.cpp
)

# Add cmake target dependencies of the executable
#add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_generate_messages_cpp)

# Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
    ${InferenceEngine_LIBRARIES} 
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
)

if(TARGET IE::ie_cpu_extension)
    target_link_libraries(${PROJECT_NAME} 
    IE::ie_cpu_extension
    )
endif()


#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}
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

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)
