/*
 * NODELET HELPER
 *
 * Copyright (c) 2020-2021 Alberto José Tudela Roldán <ajtudela@gmail.com>
 * 
 * This file is part of object_detection_openvino project.
 * 
 * All rights reserved.
 *
 */

#include <nodelet/nodelet.h>

namespace nodelet_helper{
	template<typename T>
	class TNodelet: public nodelet::Nodelet{
		public:
			TNodelet(){};

			void onInit(){
				NODELET_DEBUG("Initializing nodelet");

				nodelet_ = std::unique_ptr<T>(new T(getNodeHandle(), getPrivateNodeHandle()));
			}

		private:
			std::unique_ptr<T> nodelet_;
	};
} // End nodelet_helper namespace
