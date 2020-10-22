object_detection_openvino
========================

An implementation of YOLO and Mobilenet-SSD object detection with a ROS interface and enhanced processor utilization (40-20 fps) using OpenVINO model optimization tools. 
A depth camera is needed to perform 3d coordinates extraction.

The work is based on [this](https://github.com/PINTO0309/OpenVINO-YoloV3).

Requirements
============
* OpenVINO 2020.3 or newer.

Usage
=====

Launch the node as follows:
```
roslaunch object_detection_openvino object_detection_openvino.launch
```

Or if you can launch the node with the Intel RealSense camera:
```
roslaunch object_detection_openvino object_detection_openvino_realsense.launch
```

Parameters
----------
 * ```~model_thresh```
  [float, default:0.3]
  Detection threshold.
 * ```~model_iou_thresh```
  [float, default:0.4]
  Bounding box intersection overlaping threshold.
 * ```~model_xml```
  Filepath to the network configuration file.
 * ```~model_bin```
  Filepath to the network weigths fie.
 * ```~model_labels```
  Filepath to the network labels file.
 * ```~model_type```
  [std::string, default:YOLO]
  Network type. Can be YOLO or SSD.
 * ```~device_target```
  [std::string, default:CPU]
  Device target. Can be CPU, GPU or a MYRIAD device.
 * ```~show_fps```
  [bool, default:false]
  Option to show the fps in the image.

Subscriptions
----------
 * ```color_topic```
  [sensor_msgs/Image, default: /camera/color/image_raw]
  Color image topic where detection will be performed.
 * ```depth_topic```
  [sensor_msgs/Image, default: /camera/depth/image_raw]
  Depth image topic for extracting 3d coordinates.
 * ```info_topic```
  [sensor_msgs/CameraInfo, default: /camera/info]
  Topic for the camera information.
 
Publications
----------
 * ```image_detected_topic```
  [sensor_msgs/Image, default: detected_image]
  Image with the bounding boxes sourrunding the objects detected.
 * ```detection2d_topic```
  [object_detection_openvino/Detection2DArray, default: detection2d]
  List with the objects detected in the image.
 * ```detection3d_topic```
  [object_detection_openvino/Detection3DArray, default: detection3d]
  List with the objects detected in the color and depth images.
 * ```detection_markers```
  [visualization_msgs/MarkerArray]
  3d markers of the objects detected.
