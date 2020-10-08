# Message of the bounding box 3d for detected object in 3d coordinates

uint64 					id					# Id of the object detected
string 					Class				# Class of the object
float64 				confidence			# Probability of the object
float64 				xmin				# Upper left x corner of the bounding box in 3d coordinates
float64 				ymin				# Upper left y corner of the bounding box in 3d coordinates
float64 				xmax				# Lower right x corner of the bounding box in 3d coordinates
float64 				ymax				# Lower right y corner of the bounding box in 3d coordinates
float64 				zmin				# Near front of the bounding box in 3d coordinates
float64 				zmax				# Far rear of the bounding box in 3d coordinates
