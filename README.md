# object_detection_openvino

![ROS2](https://img.shields.io/badge/ros2-galactic-purple?logo=ros&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/license-Apache%202-blue)

## Overview

An implementation of YOLO and Mobilenet-SSD object detection with a [ROS2] interface and enhanced processor utilization using OpenVINO model optimization tools. It can be use with any Myriad X, i.e.: Intel Neural Compute Stick 2.
This package is designed on async api of [Intel OpenVINO](https://software.intel.com/en-us/openvino-toolkit) and allows an easy setup for **object detection**.<br><br>
If you provide a pointcloud image (from an Intel RealSense Camera, for example) you can obtain 3d boxes and markers in [Rviz2].

**Keywords:** ROS, ROS2, OpenVino, RealSense, OpenCV, object_detection

### License

The source code is released under a [Apache license 2.0](LICENSE).

**Author: Alberto Tudela<br />**

The object_detection_openvino package has been tested under [Rviz2] Galactic on [Ubuntu] 20.04. This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.

## Installation

### Building from Source

#### Dependencies

- [Robot Operating System (ROS)](http://wiki.ros.org) (middleware for robotics),
- [Intel OpenVINO 2021.4.1](https://software.intel.com/en-us/openvino-toolkit) (toolkit for deep learning),
- [OpenCV 3](https://opencv.org/) (Computer Vision library),
- [Boost](https://www.boost.org/) (C++ source libraries)

#### Building

To build from source, clone the latest version from this repository into your catkin workspace and compile the package using

	cd colcon_workspace/src
	git clone https://github.com/ajtudela/object_detection_openvino.git
	cd ../
	rosdep install -i --from-path src --rosdistro galactic -y
	colcon build --symlink-install

## Usage

Run the main node with:

	ros2 launch object_detection_openvino default.launch.py

## Download weights file
The weights file is very large and needs to be downloaded separately.

Download the weights file to object_detection_openvino/models/ to install it.

	cd object_detection_openvino/models
	./downloadYoloModels.sh

## Nodes

### object_detection_openvino_node

Perform object detection using OpenVino.


#### Subscribed Topics

* **`/camera/color/image_raw`** ([sensor_msgs/Image])

	Color image topic where detection will be performed.

* **`/camera/color/points`** ([sensor_msgs/PointCloud2]) (Optional)

	Registered pointcloud topic. Is this topic is not empty, it will extract a 3d bounding box of the objects detected. This pointcloud must be ordered.

#### Published Topics

* **`image_raw`** ([sensor_msgs/Image])

	Image with the bounding boxes surrounding the detected objects.

* **`detection_info`** ([vision_msgs/VisionInfo])

	Provides meta-information about the detection pipeline: method, database location,...

* **`detections2d`** ([vision_msgs/Detection2DArray])

	List with the detected objects in the image.

* **`detections3d`** ([vision_msgs/Detection3DArray])

	List with the detected objects in the image if pointcloud is enabled.

* **`markers`** ([visualization_msgs/MarkerArray])

	3d markers of the objects detected.

#### Parameters

* **`camera_frame`** (string, default: "camera_link")

	Frame that all measurements are based on.

* **`class_labels`** (string array, default: "-")

	String array with the labels of the classes.

* **`model_thresh`** (float, default: 0.3)

	Detection threshold.

* **`model_iou_thresh`** (float, default: 0.4)

	Bounding box intersection overlaping threshold.

* **`model_xml`** (string, default: "-")

	Filepath to the network configuration file.

* **`model_bin`** (string, default: "-")

	Filepath to the network weigths fie.

* **`model_labels`** (string, default: "-")

	Filepath to the network labels file.

* **`model_type`** (string, default: "YOLO")

	Network type. Can be YOLO or SSD.

* **`device_target`** (string, default: "CPU")

	Device target. Can be CPU, GPU or a MYRIAD device.

* **`show_fps`** (bool, default: false)

	Option to show the fps in the image.

## Future work
- [ ] Convert nodes to LifeCycleNodes.

[Ubuntu]: https://ubuntu.com/
[ROS2]: https://docs.ros.org/en/galactic/
[Rviz2]: https://github.com/ros2/rviz
[sensor_msgs/Image]: http://docs.ros.org/api/sensor_msgs/html/msg/Image.html
[sensor_msgs/PointCloud2]: http://docs.ros.org/api/sensor_msgs/html/msg/PointCloud2.html
[vision_msgs/VisionInfo]: http://docs.ros.org/api/vision_msgs/html/msg/VisionInfo.html
[vision_msgs/Detection2DArray]: http://docs.ros.org/api/vision_msgs/html/msg/Detection2DArray.html
[vision_msgs/Detection3DArray]: http://docs.ros.org/api/vision_msgs/html/msg/Detection3DArray.html
[visualization_msgs/MarkerArray]: http://docs.ros.org/api/visualization_msgs/html/msg/MarkerArray.html
