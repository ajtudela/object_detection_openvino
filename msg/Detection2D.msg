# Message of the detected object

uint64 							id					# Id of the object detected
string 							label				# Class of the object
float64 						confidence			# Confidence of the object
vision_msgs/BoundingBox2D 		bbox				# 2D bounding box surrounding the object
sensor_msgs/Image 				source_img			# 2D data that generated these results
