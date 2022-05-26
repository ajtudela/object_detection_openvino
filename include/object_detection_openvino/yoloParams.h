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
#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>

/**
 * @brief Implementation of the coordinates of regions with probability for each class for YOLOv3 model.
 * 
 */
class YoloParams{
	public:
		/// The number of regions.
		int num = 0;

		/// The number of classes for each region.
		int classes = 0;

		/// The number of coordinates for each region.
		int coords = 0;

		/// Codes a flattened list of pairs [width, height] that codes prior box sizes.
		std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};

		/**
		 * @brief Construct a new empty Yolo Params.
		 * 
		 */

		YoloParams(){}

		/**
		 * @brief Overloaded constructor, provided for convenience.
		 * 
		 * @param regionYolo The coordinates of regions with probability for each class.
		 */
		YoloParams(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo){
			coords = regionYolo->get_num_coords();
			classes = regionYolo->get_num_classes();
			anchors = regionYolo->get_anchors();
			auto mask = regionYolo->get_mask();
			num = mask.size();

			computeAnchors(mask);
		}

		/**
		 * @brief Overloaded constructor, provided for convenience.
		 * 
		 * @param layer The layer of the neural network.
		 */
		YoloParams(InferenceEngine::CNNLayer::Ptr layer){
			if(layer->type != "RegionYolo")
				throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");

			num = layer->GetParamAsInt("num");
			coords = layer->GetParamAsInt("coords");
			classes = layer->GetParamAsInt("classes");

			try{
				anchors = layer->GetParamAsFloats("anchors");
			}catch (...){}
			try{
				auto mask = layer->GetParamAsInts("mask");
				num = mask.size();

				computeAnchors(mask);
			}catch (...){}
		}

		/**
		 * @brief Copy constructor.
		 * 
		 * @param yoloParam Other yolo param.
		 */
		/*YoloParams(YoloParams& yoloParam){
			num = yoloParam.num;
			coords = yoloParam.coords;
			classes = yoloParam.classes;
			anchors = yoloParam.anchors;
		}*/
	private:

		/**
		 * @brief Compute the anchor regions.
		 * 
		 * @tparam T The type of the mask.
		 * @param mask The mask used to calculate the regions.
		 */
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
