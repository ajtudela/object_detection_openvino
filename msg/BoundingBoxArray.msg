# Array of BoundingBox messages

Header 				header					# Standard message header.
Header 				image_header			# Image message header. The timestamp of this header must be equal to the timestamp of the corresponding Image message.
BoundingBox[] 		bounding_boxes			# All bounding boxes of the current detection. 
