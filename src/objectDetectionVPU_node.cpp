/*
 * OBJECT DETECTION VPU ROS NODE
 *
 * Copyright (c) 2020-2022 Alberto José Tudela Roldán <ajtudela@gmail.com>
 * 
 * This file is part of object_detection_openvino project.
 * 
 * All rights reserved.
 *
 */

#include "rclcpp/rclcpp.hpp"
#include "object_detection_openvino/objectDetectionVPU.hpp"

/* Main */
int main(int argc, char** argv){
	rclcpp::init(argc, argv);

	auto node = std::make_shared<ObjectDetectionVPU>("object_detection_vpu");
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
