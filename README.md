# object_detection_openvino

## Overview

An implementation of YOLO and Mobilenet-SSD object detection with a ROS interface and enhanced processor utilization using OpenVINO model optimization tools. It can be use with any Myriad X, i.e.: Intel Neural Compute Stick 2.
This package is designed on async api of [Intel OpenVINO](https://software.intel.com/en-us/openvino-toolkit) and allows an easy setup for **object detection**.<br><br>
If you provide a depth image (from an Intel RealSense Camera, for example) you can obtain 3d boxes and markers in [RViz].

**Keywords:** ROS, OpenVino, RealSense, object_detection

### License

The source code is released under a [Apache license 2.0](LICENSE).

**Author: Alberto Tudela<br />**

The object_detection_openvino package has been tested under [ROS] Melodic on 18.04. This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.

## Installation

### Building from Source

#### Dependencies

- [Robot Operating System (ROS)](http://wiki.ros.org) (middleware for robotics),
- [Intel OpenVINO 2020.3.355](https://software.intel.com/en-us/openvino-toolkit) 

	sudo rosdep install --from-paths src

#### Building

To build from source, clone the latest version from this repository into your catkin workspace and compile the package using

	cd catkin_workspace/src
	git clone https://gitlab.com/ajtudela/object_detection_openvino.git
	cd ../
	rosdep install --from-paths . --ignore-src
	catkin_make

## Usage

Run the main node using YOLO with:

	roslaunch object_detection_openvino yolo_cpu.launch

Or if you can launch the same node with the Intel RealSense camera:

	roslaunch object_detection_openvino yolo_cpu_realsense.launch

Optionally, there're other configurations in the launch folder.

## Download weights file
The weights file is very large and needs to be downloaded separately.

Download the weights file to object_detection_openvino/models/ to install it.

	wget https://gitlab.com/ajtudela/object_detection_openvino/-/blob/melodic-devel/models/downloadYoloModels.sh

## Nodes

### object_detection_openvino

Perform object detection using OpenVino.


#### Subscribed Topics

* **`/camera/color/image_raw`** ([sensor_msgs/Image])

	Color image topic where detection will be performed.

* **`/camera/depth/image_raw`** ([sensor_msgs/Image])

	Depth image topic for extracting 3d coordinates. (Optional)

* **`/camera/info`** ([sensor_msgs/CameraInfo])

	Topic for the camera information.

#### Published Topics

* **`image_raw`** ([sensor_msgs/Image])

	Image with the bounding boxes surrounding the detected objects.

* **`detection_info`** ([vision_msgs/VisionInfo])

	Provides meta-information about the detection pipeline: method, database location,...

* **`detections`** ([vision_msgs/Detection2DArray] or [vision_msgs/Detection3DArray])

	List with the detected objects in the image.

* **`markers`** ([visualization_msgs/MarkerArray])

	3d markers of the objects detected.

#### Parameters

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

* **`output_image`** (bool, default: false)

	Output image of the detected objects.


[ROS]: http://www.ros.org
[Rviz]: http://wiki.ros.org/rviz
[sensor_msgs/Image]: http://docs.ros.org/api/sensor_msgs/html/msg/Image.html
[sensor_msgs/CameraInfo]: http://docs.ros.org/api/sensor_msgs/html/msg/CameraInfo.html
[vision_msgs/VisionInfo]: http://docs.ros.org/api/vision_msgs/html/msg/VisionInfo.html
[vision_msgs/Detection2DArray]: http://docs.ros.org/api/vision_msgs/html/msg/Detection2DArray.html
[vision_msgs/Detection3DArray]: http://docs.ros.org/api/vision_msgs/html/msg/Detection3DArray.html
[visualization_msgs/MarkerArray]: http://docs.ros.org/api/visualization_msgs/html/msg/MarkerArray.html
