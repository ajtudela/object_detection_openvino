/*
 * DETECTION OBJECT STRUCT
 *
 * Copyright (c) 2020-2021 Alberto José Tudela Roldán <ajtudela@gmail.com>
 * 
 * This file is part of object_detection_openvino project.
 * 
 * All rights reserved.
 *
 */

#ifndef DETECTION_OBJECT_H
#define DETECTION_OBJECT_H

#include <string>

struct DetectionObject{
	int xmin, ymin, xmax, ymax, classId;
	float confidence;
	std::string className;

	DetectionObject(double x, double y, double h, double w, int classId, std::string className, float confidence, float h_scale, float w_scale){
		xmin = static_cast<int>((x - w / 2) * w_scale);
		ymin = static_cast<int>((y - h / 2) * h_scale);
		xmax = static_cast<int>(this->xmin + w * w_scale);
		ymax = static_cast<int>(this->ymin + h * h_scale);
		this->confidence = confidence;
		this->classId = classId;
		this->className = className;
	}

	DetectionObject(double x, double y, double h, double w, int classId, std::string className, float confidence){
		xmin = static_cast<int>(x);
		ymin = static_cast<int>(y);
		xmax = static_cast<int>(this->xmin + w);
		ymax = static_cast<int>(this->ymin + h);
		this->confidence = confidence;
		this->classId = classId;
		this->className = className;
	}

	bool operator<(const DetectionObject &s2) const {
		return this->confidence < s2.confidence;
	}

	bool operator>(const DetectionObject &s2) const {
		return this->confidence > s2.confidence;
	}
};

#endif
