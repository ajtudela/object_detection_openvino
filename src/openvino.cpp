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

// C++
#include <iostream>
#include <iterator>
#include <boost/filesystem.hpp>

// Openvino
#include <ngraph/ngraph.hpp>

// ROS
#include <ros/ros.h>

#include "object_detection_openvino/openvino.h"

Openvino::Openvino(){
}

Openvino::~Openvino(){
}

void Openvino::setTargetDevice(std::string device){
	ROS_INFO_STREAM("[Object detection VPU]: Loading Inference Engine" << InferenceEngine::GetInferenceEngineVersion());
	ROS_INFO_STREAM("[Object detection VPU]: Device info: " << core_.GetVersions(device));

	// Load extensions for the plugin 
#ifdef WITH_EXTENSIONS
	if (device.find("CPU") != std::string::npos){
		/**
		 * cpu_extensions library is compiled from "extension" folder containing
		 * custom MKLDNNPlugin layer implementations. These layers are not supported
		 * by mkldnn, but they can be useful for inferring custom topologies.
		**/
		core_.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
	}else if (device.find("GPU") != std::string::npos){
		core_.SetConfig({{PluginConfigParams::KEY_DUMP_KERNELS, PluginConfigParams::YES}}, "GPU");
	}
#endif
}

void Openvino::setNetworkModel(std::string modelFileName, std::string binFileName, std::string labelFileName){
	// Read network model
	ROS_INFO("[Object detection VPU]: Loading network files");
	if (!boost::filesystem::exists(modelFileName)){
		ROS_FATAL("[Object detection VPU]: Network file doesn't exist.");
		ros::shutdown();
	}
	cnnNetwork_ = core_.ReadNetwork(modelFileName, binFileName);
	// Read labels (if any)
	std::ifstream inputFile(labelFileName);
	std::copy(std::istream_iterator<std::string>(inputFile), std::istream_iterator<std::string>(), std::back_inserter(this->labels_));
}

void Openvino::configureNetwork(std::string networkType){
	networkType_ = networkType;

	// Prepare input blobs
	ROS_INFO("[Object detection VPU]: Checking that the inputs are as expected");

	inputInfo_ = InferenceEngine::InputsDataMap(cnnNetwork_.getInputsInfo());
	if (networkType_ == "YOLO"){
		if (inputInfo_.size() != 1){
			ROS_FATAL("[Object detection VPU]: Only accepts networks that have only one input");
			ros::shutdown();
		}
		InferenceEngine::InputInfo::Ptr& input = inputInfo_.begin()->second;
		inputName_ = inputInfo_.begin()->first;
		input->setPrecision(InferenceEngine::Precision::U8);
		input->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
	}else if (networkType_ == "SSD"){
		if (inputInfo_.size() != 1 && inputInfo_.size() != 2 ){
			ROS_FATAL("[Object detection VPU]: Only accepts networks with 1 or 2 inputs");
			ros::shutdown();
		}
		for (auto &input : inputInfo_){
			// First input contains images
			if (input.second->getTensorDesc().getDims().size() == 4){
				inputName_ = input.first;
				input.second->setPrecision(InferenceEngine::Precision::U8);
				input.second->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
			// Second input contains image info
			}else if (input.second->getTensorDesc().getDims().size() == 2){
				inputName_ = input.first;
				input.second->setPrecision(InferenceEngine::Precision::FP32);
			}else{
				throw std::logic_error("[Object detection VPU]: Unsupported " + std::to_string(input.second->getTensorDesc().getDims().size()) + "D "
										"input layer '" + input.first + "'. Only 2D and 4D input layers are supported");
				ros::shutdown();
			}
		}
	}

	// Set batch size to 1
	ROS_INFO("[Object detection VPU]: Batch size is forced to  1");
	InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnnNetwork_.getInputShapes();
	InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;
	inSizeVector[0] = 1; 
	cnnNetwork_.reshape(inputShapes);

	// Prepare output blobs
	ROS_INFO("[Object detection VPU]: Checking that the outputs are as expected");
	outputInfo_ = InferenceEngine::OutputsDataMap(cnnNetwork_.getOutputsInfo());
	if (networkType_ == "YOLO"){
		if (outputInfo_.size() != 3 && outputInfo_.size() != 2){
			ROS_FATAL("[Object detection VPU]: Only accepts networks with three (YOLO) or two (tiny-YOLO) outputs");
			ros::shutdown();
		}

		for (auto &output : outputInfo_){
			output.second->setPrecision(InferenceEngine::Precision::FP32);
			output.second->setLayout(InferenceEngine::Layout::NCHW);
		}
	}else if (networkType_ == "SSD"){
		if (outputInfo_.size() != 1){
			throw std::logic_error("[Object detection VPU]: Only accepts networks with one output");
		}

		for (auto &output : outputInfo_){
			output.second->setPrecision(InferenceEngine::Precision::FP32);
			output.second->setLayout(InferenceEngine::Layout::NCHW);
		}
	}
}

void Openvino::loadModelToDevice(std::string device){
	ROS_INFO("[Object detection VPU]: Loading model to the device");
	infNetwork_ = core_.LoadNetwork(cnnNetwork_, device);
}

void Openvino::createAsyncInferRequest(){
	ROS_INFO("[Object detection VPU]: Create infer request");
	asyncInferRequestCurr_ = infNetwork_.CreateInferRequestPtr();
	asyncInferRequestNext_ = infNetwork_.CreateInferRequestPtr();
}

void Openvino::startNextAsyncInferRequest(){
	asyncInferRequestNext_->StartAsync();
}

void Openvino::swapAsyncInferRequest(){
	asyncInferRequestCurr_.swap(asyncInferRequestNext_);
}

bool Openvino::isDeviceReady(){
	return InferenceEngine::OK == asyncInferRequestCurr_->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
}

std::vector<std::string> Openvino::getLabels(){
	return labels_;
}

std::vector<DetectionObject> Openvino::getDetectionObjects(size_t height, size_t width, float threshold, float iouThreshold){
	// Processing output blobs of the CURRENT request
	const InferenceEngine::TensorDesc& inputDesc = inputInfo_.begin()->second.get()->getTensorDesc();
	unsigned long resizedImgH = getTensorHeight(inputDesc);
	unsigned long resizedImgW = getTensorWidth(inputDesc);

	// Parsing outputs
	std::vector<DetectionObject> objects;
	for (auto &output: outputInfo_){
		auto outputName = output.first;
		InferenceEngine::Blob::Ptr blob = asyncInferRequestCurr_->GetBlob(outputName);

		if (networkType_ == "YOLO") parseYOLOV3Output(cnnNetwork_, outputName, blob, resizedImgH, resizedImgW, height, width, threshold, objects);
		else if (networkType_ == "SSD") parseSSDOutput(blob, height, width, threshold, objects);
	}

	// Filtering overlapping boxes
	std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
	for (int i = 0; i < objects.size(); ++i){
		if (objects[i].confidence == 0)
			continue;
		for (int j = i + 1; j < objects.size(); ++j) {
			if (intersectionOverUnion(objects[i], objects[j]) >= iouThreshold) {
				objects[j].confidence = 0;
			}
		}
	}

	return objects;
}

void Openvino::frameToNextInfer(const cv::Mat &frame, bool autoResize){
	frameToBlob(frame, asyncInferRequestNext_, inputName_, autoResize);
}

int Openvino::entryIndex(int side, int lcoords, int lclasses, int location, int entry) {
	int n = location / (side * side);
	int loc = location % (side * side);
	return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

double Openvino::intersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
	double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
	double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
	double area_of_overlap;

	if (width_of_overlap_area < 0 || height_of_overlap_area < 0){
		area_of_overlap = 0;
	}else{
		area_of_overlap = width_of_overlap_area * height_of_overlap_area;
	}

	double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
	double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
	double area_of_union = box_1_area + box_2_area - area_of_overlap;

	return area_of_overlap / area_of_union;
}

void Openvino::frameToBlob(const cv::Mat &frame, InferenceEngine::InferRequest::Ptr &inferRequest, const std::string &inputName, bool autoResize){
	if (autoResize){
		// Just set input blob containing read image. Resize and layout conversion will be done automatically 
		inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
	}else{
		// Resize and copy data from the image to the input blob
		InferenceEngine::Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
		matU8ToBlob<uint8_t>(frame, frameBlob);
	}
}

void Openvino::parseSSDOutput(const InferenceEngine::Blob::Ptr &blob, const unsigned long height, const unsigned long width, const float threshold, std::vector<DetectionObject> &objects){
	// Validating output parameters
	InferenceEngine::SizeVector outputDims = blob->getTensorDesc().getDims();
	int maxProposalCount = static_cast<int>(blob->getTensorDesc().getDims()[2]);
	const int objectSize = static_cast<int>(blob->getTensorDesc().getDims()[3]);

	if (objectSize != 7){
		throw std::logic_error("[Object detection VPU]: Output should have 7 as a last dimension");
	}
	if (outputDims.size() != 4){
		throw std::logic_error("[Object detection VPU]: Incorrect output dimensions for SSD");
	}

	const float *outputBlob = blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
	for (int i = 0; i < maxProposalCount; i++){
		float id = outputBlob[i * objectSize + 0];

		if (id < 0) break;

		auto label = static_cast<int>(outputBlob[i * objectSize + 1]);
		float prob = outputBlob[i * objectSize + 2];
		float xmin = outputBlob[i * objectSize + 3] * width;
		float ymin = outputBlob[i * objectSize + 4] * height;
		float xmax = outputBlob[i * objectSize + 5] * width;
		float ymax = outputBlob[i * objectSize + 6] * height;

		double newWidth = xmax - xmin;
		double newHeight = ymax - ymin;

		if (prob < threshold) continue;

		DetectionObject obj(xmin, ymin, newHeight, newWidth, label, this->labels_[label], prob);
		objects.push_back(obj);
	}
}

void Openvino::parseYOLOV3Output(const InferenceEngine::CNNNetwork &cnnNetwork, const std::string &outputName, const InferenceEngine::Blob::Ptr &blob, const unsigned long resizedImgH, const unsigned long resizedImgW, const unsigned long originalImgH, const unsigned long originalImgW, const float threshold, std::vector<DetectionObject> &objects) {
	// Validating output parameters 
	const int outBlobH = static_cast<int>(blob->getTensorDesc().getDims()[2]);
	const int outBlobW = static_cast<int>(blob->getTensorDesc().getDims()[3]);
	if (outBlobH != outBlobW){
		throw std::runtime_error("[Object detection VPU]: Invalid size of output " + outputName +
		" It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(outBlobH) +
		", current W = " + std::to_string(outBlobW));
	}

	// Extracting layer parameters
	YoloParams params;
	if (auto ngraphFunction = cnnNetwork.getFunction()){
		for (const auto op: ngraphFunction->get_ops()){
			if (op->get_friendly_name() == outputName){
				auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
				if (!regionYolo){
					throw std::runtime_error("Invalid output type: " +
					std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
				}

				params = regionYolo;
				break;
			}
		}
	}else{
		throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
	}

	auto side = outBlobH;
	auto sideSquare = side * side;
	const float *outputBlob = blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();

	// Parsing YOLO Region output 
	for (int i = 0; i < sideSquare; ++i){
		int row = i / side;
		int col = i % side;
		for (int n = 0; n < params.num; ++n){
			int objIdx = entryIndex(side, params.coords, params.classes, n * side * side + i, params.coords);
			int boxIdx = entryIndex(side, params.coords, params.classes, n * side * side + i, 0);
			float scale = outputBlob[objIdx];

			if (scale < threshold) continue;

			double x = (col + outputBlob[boxIdx + 0 * sideSquare]) / side * resizedImgW;
			double y = (row + outputBlob[boxIdx + 1 * sideSquare]) / side * resizedImgH;
			double height = std::exp(outputBlob[boxIdx + 3 * sideSquare]) * params.anchors[2 * n + 1];
			double width = std::exp(outputBlob[boxIdx + 2 * sideSquare]) * params.anchors[2 * n];
			for (int j = 0; j < params.classes; ++j) {
				int classIdx = entryIndex(side, params.coords, params.classes, n * sideSquare + i, params.coords + 1 + j);
				float prob = scale * outputBlob[classIdx];

				if (prob < threshold) continue;

				DetectionObject obj(x, y, height, width, j, this->labels_[j], prob,
					static_cast<float>(originalImgH) / static_cast<float>(resizedImgH),
					static_cast<float>(originalImgW) / static_cast<float>(resizedImgW));
				objects.push_back(obj);
			}
		}
	}
}
