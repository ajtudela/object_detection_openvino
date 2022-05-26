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

/**
 * @brief Implementation of a detection object in an image
 * 
 * An object is defined by the 2D coordinates in pixels of the bounding box surrounding it,
 * its class (name and numeric id) and its confidence.
 */
struct DetectionObject{
	/// Minimum coordinate in the X axis.
	int xmin;

	/// Minimum coordinate in the Y axis.
	int ymin;

	/// Maximum coordinate in the X axis.
	int xmax;

	/// Maximum coordinate in the Y axis.
	int ymax;

	/// Numeric id of the object class.
	int classId;

	/// Confidence of the detected object in a range between 0 and 1.
	float confidence;

	/// Name of object class.
	std::string className;

	/**
	 * @brief Constructor of a new detection object.
	 * 
	 * @param x Coordinate of the center pixel in the X axis. 
	 * @param y Coordinate of the center pixel in the Y axis. 
	 * @param h Height in pixels of the bounding box surrounding the detected object.
	 * @param w Width in pixels of the bounding box surrounding the detected object.
	 * @param classId  Numeric id of the object class.
	 * @param className Name of object class.
	 * @param confidence Confidence of the detected object in a range between 0 and 1.
	 * @param h_scale Scale of the bounding box height.
	 * @param w_scale Scale of the bounding box width.
	 */
	DetectionObject(double x, double y, double h, double w, int classId, std::string className, float confidence, float h_scale, float w_scale){
		xmin = static_cast<int>((x - w / 2) * w_scale);
		ymin = static_cast<int>((y - h / 2) * h_scale);
		xmax = static_cast<int>(this->xmin + w * w_scale);
		ymax = static_cast<int>(this->ymin + h * h_scale);
		this->confidence = confidence;
		this->classId = classId;
		this->className = className;
	}

	/**
	 * @brief Overloaded constructor, provided for convenience.
	 * It differs from the above function only in what argument(s) it accepts.
	 * 
	 * @param x Coordinate of the center pixel in the X axis. 
	 * @param y Coordinate of the center pixel in the Y axis. 
	 * @param h Height in pixels of the bounding box surrounding the detected object.
	 * @param w Width in pixels of the bounding box surrounding the detected object.
	 * @param classId  Numeric id of the object class.
	 * @param className Name of object class.
	 * @param confidence Confidence of the detected object in a range between 0 and 1.
	 */
	DetectionObject(double x, double y, double h, double w, int classId, std::string className, float confidence){
		xmin = static_cast<int>(x);
		ymin = static_cast<int>(y);
		xmax = static_cast<int>(this->xmin + w);
		ymax = static_cast<int>(this->ymin + h);
		this->confidence = confidence;
		this->classId = classId;
		this->className = className;
	}

	/**
	 * @brief Copy constructor.
	 * 
	 * @param object The detected object.
	 */
	DetectionObject(DetectionObject& object){
		xmin = object.xmin;
		ymin = object.ymin;
		xmax = object.xmax;
		ymax = object.ymax;
		confidence = object.confidence;
		classId = object.classId;
		className = object.className;
	}

	/**
	 * @brief Copy constructor.
	 * 
	 * @param object The detected object.
	 */
	DetectionObject(const DetectionObject& object){
		xmin = object.xmin;
		ymin = object.ymin;
		xmax = object.xmax;
		ymax = object.ymax;
		confidence = object.confidence;
		classId = object.classId;
		className = object.className;
	}

	/**
	 * @brief Relational operator of the confidence of the detected object.
	 * 
	 * @param s2 Object to compare to. 
	 * @return True if the left object has lower confidence than the right one. False, otherwise.
	 */
	bool operator<(const DetectionObject &s2) const {
		return this->confidence < s2.confidence;
	}

	/**
	 * @brief Relational operator of the confidence of the detected object.
	 * 
	 * @param s2 Object to compare to. 
	 * @return True if the left object has greater confidence than the right one. False, otherwise.
	 */
	bool operator>(const DetectionObject &s2) const {
		return this->confidence > s2.confidence;
	}
};

#endif
