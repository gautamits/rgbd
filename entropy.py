#import matplotlib.pyplot as plt
import numpy as np

#from skimage import data
import matplotlib.pyplot as plt
import cv2
from skimage.util import img_as_ubyte
from skimage.filter.rank import entropy
from skimage.morphology import disk


# First example: object detection.
def entro(image):
    if len(image.shape) > 2:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    """noise_mask = 28 * np.ones((128, 128), dtype=np.uint8)
    noise_mask[32:-32, 32:-32] = 30

    noise = (noise_mask * np.random.random(noise_mask.shape) - 0.5 *
             noise_mask).astype(np.uint8)
    img = noise + 128

    entr_img = entropy(img, disk(10))
	"""
    
    entr_img=entropy(image, disk(4))
    #normalisation
    peak=max(entr_img.flatten())
    #print entropy
    entr_img=(255/peak)*entr_img
    return entr_img
if __name__=="__main__":
    i=cv2.imread("images/amit1.jpg")
    gray=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
    #print i
    entropy=entro(gray)
    
    print np.histogram(entropy,range(64))[0]
    plt.imshow(entropy,cmap="gray")
    plt.show()
