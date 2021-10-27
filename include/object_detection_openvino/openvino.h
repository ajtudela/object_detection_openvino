/*
 * OPENVINO CLASS
 *
 * Copyright (c) 2020-2021 Alberto José Tudela Roldán <ajtudela@gmail.com>
 * 
 * This file is part of object_detection_openvino project.
 * 
 * All rights reserved.
 *
 */

#ifndef OPENVINO_H
#define OPENVINO_H

// C++
#include <map>
#include <string>
#include <vector>

// OpenVINO
#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>

#ifdef WITH_EXTENSIONS
    #include <ext_list.hpp>
#endif

// Object detection
#include <object_detection_openvino/detectionObject.h>
#include <object_detection_openvino/yoloParams.h>

class Openvino{
	public:
		Openvino();
		~Openvino();
		void setTargetDevice(std::string device);
		void setNetworkModel(std::string modelFileName, std::string binFileName, std::string labelFileName);
		void configureNetwork(std::string networkType);
		void loadModelToDevice(std::string device);
		void createAsyncInferRequest();
		void startNextAsyncInferRequest();
		void swapAsyncInferRequest();
		bool isDeviceReady();
		std::vector<std::string> getLabels();
		std::vector<DetectionObject> getDetectionObjects(size_t height, size_t width, float iouThreshold);
		void frameToNextInfer(const cv::Mat &frame, bool autoResize = false);

	private:
		float thresh_;
		std::string inputName_, networkType_, deviceTarget_;
		std::map<std::string, YoloParams> yoloParams_;
		InferenceEngine::ExecutableNetwork infNetwork_;
		InferenceEngine::InferRequest::Ptr asyncInferRequestCurr_, asyncInferRequestNext_;
		InferenceEngine::OutputsDataMap outputInfo_;
		InferenceEngine::InputsDataMap inputInfo_;
		InferenceEngine::CNNNetwork cnnNetwork_;
		InferenceEngine::Core core_;
		std::vector<std::string> labels_;

		static int entryIndex(int side, int lcoords, int lclasses, int location, int entry);
		double intersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2);
		void frameToBlob(const cv::Mat &frame, InferenceEngine::InferRequest::Ptr &inferRequest, const std::string &inputName, bool autoResize = false);
		void parseSSDOutput(const InferenceEngine::Blob::Ptr &blob, const unsigned long height, const unsigned long width, const float threshold, std::vector<DetectionObject> &objects);
		void parseYOLOV3Output(const InferenceEngine::CNNNetwork &cnnNetwork, const std::string &outputName, const InferenceEngine::Blob::Ptr &blob, const unsigned long resizedImgH, const unsigned long resizedImgW, const unsigned long originalImgH, const unsigned long originalImgW, const float threshold, std::vector<DetectionObject> &objects);
};
#endif
