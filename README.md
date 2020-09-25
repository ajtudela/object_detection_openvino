object_detection_openvino
========================

An implementation of YOLO and Mobilenet-SSD object detection with a ROS interface and enhanced processor utilization (40-20 fps) using OpenVINO model optimization tools. A depth camera is needed to perform 3d coordinates extraction.

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
  Network type. cAn be YOLO or SSD.
 * ```~device_target```
  [std::string, default:CPU]
  Device target. Can be CPU, GPU or a MYRIAD device.
 * ```~show_fps```
  [bool, default:false]
  Option to show the fps in the image.

Subscriptions
----------
 * ```color_topic```
  [sensor_msgs/Image]
  Color image topic where detection will be performed.
 * ```color_topic```
  [sensor_msgs/Image]
  Depth image topic for extracting 3d coordinates.
 * ```color_topic```
  [sensor_msgs/CameraInfo]
  Topic for the camera information.
 
Publications
----------
 * ```detection_image```
  [sensor_msgs/Image]
  Image with the bounding boxes sourrunding the objects detected.
 * ```bounding_boxes```
  [object_detection_openvino/BoundingBoxArray]
  List with the bounding boxes sourrunding the objects detected.
 * ```bounding_boxes3d```
  [object_detection_openvino/BoundingBox3dArray]
  List with the 3d bounding boxes sourrunding the objects detected.
 * ```detection_markers```
  [visualization_msgs/MarkerArray]
  3d markers of the objects detected.
