# Message of the detected object

uint64 							id					# Id of the object detected
string 							label				# Class of the object
float64 						confidence			# Confidence of the object
vision_msgs/BoundingBox3D 		bbox				# 3D bounding box surrounding the object
sensor_msgs/PointCloud2 		source_cloud		# 3D data that generated these results
