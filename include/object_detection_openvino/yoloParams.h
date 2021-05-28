/*
 * YOLO PARAMS CLASS
 *
 * Copyright (c) 2020-2021 Alberto José Tudela Roldán <ajtudela@gmail.com>
 * 
 * This file is part of object_detection_openvino project.
 * 
 * All rights reserved.
 *
 */
 
#ifndef YOLO_PARAMS_H
#define YOLO_PARAMS_H

// OpenVINO
#include <ngraph/ngraph.hpp>

class YoloParams{
	public:
		int num = 0, classes = 0, coords = 0;
		std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};

		YoloParams(){}

		YoloParams(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo){
			coords = regionYolo->get_num_coords();
			classes = regionYolo->get_num_classes();
			anchors = regionYolo->get_anchors();
			auto mask = regionYolo->get_mask();
			num = mask.size();

			computeAnchors(mask);
		}
	private:
		template <typename T> void computeAnchors(const std::vector<T> & mask){
			std::vector<float> maskedAnchors(num * 2);
			for (int i = 0; i < num; ++i) {
				maskedAnchors[i * 2] = anchors[mask[i] * 2];
				maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
			}
			anchors = maskedAnchors;
		}
};

#endif
