from PIL import Image
import numpy as np
import cv2

   
# load our serialized black and white colorizer model and cluster
# center points from disk
    
net = cv2.dnn.readNetFromCaffe("model/colorization_deploy_v2.prototxt", "model/colorization_release_v2.caffemodel")
pts = np.load("model/pts_in_hull.npy")
# add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
# it will load the image from disk and scale the intensities of each pixel to the range [0, 1]
image = cv2.imread(myimage)
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
resized = cv2.resize(lab, (224, 224))# resize the Lab image to 224x224 
L = cv2.split(resized)[0]
L -= 50
# pass the L channel through the network which will *predict* the 'a'
# and 'b' channel values
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))#resize the predicted 'ab' volume
# grab the 'L' channel from the *original* image
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR) # convert the output image to RGB
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")
colorized = cv2.resize(colorized, (300, 350))
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
