import cv2
import numpy as np
import os.path


class NormalDepthCalc():




	def normalFromDepth(image):


		# Draw depth image
		colorNormal = self.drawDepth(processedDisparity)


		return colorNormal
        
        
    
    def depth_to_normal(y_pred):
      zy, zx = tf.image.image_gradients(y_pred)    
      normal_ori = tf.concat([-zx, -zy, tf.ones_like(y_pred)], 3) 
      new_normal = tf.square(zx) +  tf.square(zy) + 1
      normal = normal_ori/new_normal
      normal += 1
      normal /= 2

      return normal
      
    def compute_normals(elevation):
        height, width, nchan = elevation.shape
        assert nchan == 1
        normals = np.empty([height - 1, width - 1, 3])
        _compute_normals(elevation[:,:,0], normals)
        return normals 

