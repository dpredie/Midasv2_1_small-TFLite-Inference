import cv2
import numpy as np
from MidasDepthEstimation.midasDepthEstimator import midasDepthEstimator
from NormalsFromDepth.NormalDepthCalc import NormalDepthCalc
import time
import argparse

inputfile = "/content/inputs/input.mp4"
outputfolder= "/content/results"
fps=24.0
opt = 1

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--output_file', type=str, required=True, help='copy path from output directory')
parser.add_argument('--input_file', type=str, required=True, help='copy path from input file')
parser.add_argument('--fps', type=float, required=True, help='fps value')
parser.add_argument('--out_opt', type=int, help='1 = output depth; 2= output RGB+depth; 3=normals')



# Check for required input
option_, _ = parser.parse_known_args()

#if option_.out_opt:
#  opt = option_.out_opt


print(option_)
outputfile = option_.output_file
inputfile = option_.input_file
fps = option_.fps


tile =1

if option_.out_opt == 0:
  opt = option_.out_opt
  tile=1
elif option_.out_opt == 1:
  opt = option_.out_opt
  tile=1
elif option_.out_opt ==2:
  opt = option_.out_opt
  tile=2
#elif option_.out_opt ==3:
#  opt = option_.out_opt
#  tile=1

print('Option: '+str(opt))

# Initialize depth estimation model
depthEstimator = midasDepthEstimator()

# Initialize video
cap = cv2.VideoCapture(inputfile)

print(cap)
out = cv2.VideoWriter(outputfile,cv2.VideoWriter_fourcc(*'MP4V'), fps,(int(cap.get(3)*tile),int(cap.get(4))))

if (cap.isOpened()== False):
  print("Error opening video stream or file")

framecnt = 0
while cap.isOpened():

  # Read frame from the video
  ret, img = cap.read()

  if ret:  
    start = time.time()
    #now = datetime.datetime.now()
    # Estimate depth
    colorDepth = depthEstimator.estimateDepth(img)

 #   colorNormal = NormalDepthCalc.normalFromDepth(colorDepth)
    # Add the depth image over the color image:
    #combinedImg = cv2.addWeighted(img,0.7,colorDepth,0.6,0)

    # Join the input image, the estiamted depth and the combined image
    
    if opt==0:
      img_out = img
    elif opt<3:
      if tile==2:
        img_out = np.hstack((img, colorDepth))
      elif tile==1: 
        img_out = colorDepth
 #   else:
 #       img_out = colorNormal


    #print(img_out)
    #cv2.imshow("Depth Image", img_out)
    out.write(img_out)
    end = time.time()
    print("processed frame: "+str(framecnt)+" "+str("%.2f" % (end-start))+"s")
    framecnt+=1
  else:
    print("image empty - exiting")    
    break  
  # Press key q to stop
  if cv2.waitKey(1) == ord('q'):
    break
  


cap.release()
out.release()
cv2.destroyAllWindows()
