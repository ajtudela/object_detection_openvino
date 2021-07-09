/*
 * OBJECT DETECTION OPENVINO ROS NODE
 *
 * Copyright (c) 2020-2021 Alberto José Tudela Roldán <ajtudela@gmail.com>
 * 
 * This file is part of object_detection_openvino project.
 * 
 * All rights reserved.
 *
 */

#include <ros/ros.h>
#include <object_detection_openvino/objectDetectionOpenvino.h>

/* Main */
int main(int argc, char** argv){
	ros::init(argc, argv, "object_detection_openvino");
	ros::NodeHandle node("");
	ros::NodeHandle node_private("~");

	try{
		ROS_INFO("[Object detection Openvino]: Initializing node");
		ObjectDetectionOpenvino detector(node, node_private);
		ros::spin();
	}catch(const char* s){
		ROS_FATAL_STREAM("[Object detection Openvino]: " << s);
	}catch(...){
		ROS_FATAL_STREAM("[Object detection Openvino]: Unexpected error");
	}
}
