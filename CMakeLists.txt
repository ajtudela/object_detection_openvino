cmake_minimum_required(VERSION 3.8)
project(object_detection_openvino)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

################################################
## Find  dependencies                         ##
################################################
## Find ament macros and libraries
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Boost REQUIRED)

# Ngraph and InferenceEngine
find_package(ngraph REQUIRED)
find_package(InferenceEngine 2021.4.1 REQUIRED)
if(NOT(InferenceEngine_FOUND))
    message(WARNING "InferenceEngine is disabled or not found.")
    return()
else()
    include_directories(${InferenceEngine_INCLUDE_DIRS}/../samples/cpp/common/)
    include_directories(${InferenceEngine_INCLUDE_DIRS}/../samples/cpp/common/utils/include/)
endif()

if(TARGET IE::ie_cpu_extension)
    add_definitions(-DWITH_EXTENSIONS)
endif()

###########
## Build ##
###########
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    include
    ${Boost_INCLUDE_DIRS}
    ${InferenceEngine_INCLUDE_DIRS}
)

# Set SOURCES
set(SOURCES
    src/objectDetectionVPU.cpp
    src/openvino.cpp
)

# Executable node
add_executable (${PROJECT_NAME}
    ${SOURCES}
    src/objectDetectionVPU_node.cpp
)
ament_target_dependencies(${PROJECT_NAME}
    rclcpp
    tf2
    std_msgs 
    sensor_msgs
    vision_msgs
    image_transport
    message_filters
    visualization_msgs
    cv_bridge
    pcl_ros
    pcl_conversions
)

# Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
    ${InferenceEngine_LIBRARIES}
    ${NGRAPH_LIBRARIES}
    ${Boost_LIBRARIES}
)

if(TARGET IE::ie_cpu_extension)
    target_link_libraries(${PROJECT_NAME} IE::ie_cpu_extension)
endif()


#############
## Install ##
#############
install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION include/${PROJECT_NAME}
    FILES_MATCHING PATTERN "*.hpp"
    #   PATTERN ".svn" EXCLUDE
)

install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY models
    DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY params
    DESTINATION share/${PROJECT_NAME}/
)

#############
## Testing ##
#############
if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
