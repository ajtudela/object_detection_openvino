^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package object_detection_openvino
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4.0.0 (13-12-2022)
------------------
* First ROS2 (Galactic) version.
* Update to OpenVino 2021.4.

3.2.2 (26-05-2022)
------------------
* Improve documentation.
* Fix threshold value not passing to openvino.

3.2.1 (08-03-2022)
------------------
* Optimizations, change pose to bottom of bbox.

3.2.0 (23-02-2022)
------------------
* Fix for empty objects in messages when NaN values in center position.
* Remove output_image parameter.
* Add filter to realsense launch files.

3.1.0 (15-02-2022)
------------------
* Fix frames of detections2D and detections3D.
* Only publish if objects are detected.
* Change database_location in info message from filename to location in param server. Defined as "class_labels".

3.0.1 (03-02-2022)
------------------
* Add object pose in detection3d.

3.0.0 (14-01-2022)
------------------
* Use pointcloud instead of depth topic to increase accuracy.
* Split openvino into a different class to improve code readibility.
* Publish both detections2D and detections3D when using depth instead of detections3D only.
* Change frame of detections to camera frame.

2.0.1 (23-11-2021)
------------------
* Added Orbbec Astra Stereo S USB 3.0 launch file.

2.0.0 (13-09-2021)
------------------
* Create CHANGELOG.rst.
* Fix camera info in depth analysis.

1.6.0 (09-07-2021)
------------------
* Added nodelet option for vision pipeline.

1.5.0 (25-06-2021)
------------------
* Change to OpenVino 2020.3.355.
* Remove custom messages and replaces them by vision_msgs.

1.1.0 (21-10-2020)
------------------
* Update OpenVino version to 2020.3

1.0.0 (25-09-2020)
------------------
* Initial release.
* Create README.md.
* Added objectDetectionOpenvino class (.h and .cpp files).
* Added BoundingBox.msg, BoundingBox3d.msg, BoundingBox3dArray.msg and BoundingBoxArray.msg messages.
* Added launch files.
* Added initial YOLO and mobilenet-ssd models.
* Contributors: Alberto Tudela
