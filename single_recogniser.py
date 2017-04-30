import cv2
import easygui
import numpy as np
from matcher import euclidean_matcher
color_path=easygui.fileopenbox("choose color image")

rgb=cv2.imread(color_path)
rgb=cv2.resize(rgb,(140,140))
depth_path=color_path.split("/")[::-1]
depth_path[1]="Depth"
depth_path=depth_path[::-1]
depth_path="/".join(depth_path)
print depth_path
depth=cv2.imread(depth_path)
depth=cv2.resize(depth,(140,140))
label=euclidean_matcher(rgbd(rgb,depth))
naming=np.load("database/naming.npy")
print naming[label],
#print labels[:10]

