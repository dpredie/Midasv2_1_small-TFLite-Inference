import cv2
import numpy as np
import os.path


class NormalDepthCalc():

  def normalFromDepth(image):
    # Draw depth image - https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python
    d_im = image
    d_im = d_im.astype("float64")
    zy, zx, zz = np.gradient(d_im)  
    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise
    #zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)     
    #zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    normal *= 255


    colorNormal = normal[:, :, ::-1]
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

