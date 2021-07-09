/*
 * OBJECT DETECTION OPENVINO NODELET
 *
 * Copyright (c) 2020-2021 Alberto José Tudela Roldán <ajtudela@gmail.com>
 * 
 * This file is part of object_detection_openvino project.
 * 
 * All rights reserved.
 *
 */

#include <pluginlib/class_list_macros.h>
#include <object_detection_openvino/objectDetectionOpenvino.h>
#include <object_detection_openvino/objectDetectionOpenvino_nodelet.h>

namespace object_detection_openvino{
	using ObjectDetectionOpenvinoNodelet = nodelet_helper::TNodelet<ObjectDetectionOpenvino>;
}

PLUGINLIB_EXPORT_CLASS(object_detection_openvino::ObjectDetectionOpenvinoNodelet, nodelet::Nodelet)
