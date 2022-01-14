/*
 * OBJECT DETECTION VPU NODELET
 *
 * Copyright (c) 2020-2021 Alberto José Tudela Roldán <ajtudela@gmail.com>
 * 
 * This file is part of object_detection_openvino project.
 * 
 * All rights reserved.
 *
 */

#include <pluginlib/class_list_macros.h>
#include <object_detection_openvino/objectDetectionVPU.h>
#include <object_detection_openvino/nodeletHelper.h>

namespace object_detection_openvino{
	using ObjectDetectionVPUNodelet = nodelet_helper::TNodelet<ObjectDetectionVPU>;
}

PLUGINLIB_EXPORT_CLASS(object_detection_openvino::ObjectDetectionVPUNodelet, nodelet::Nodelet)
