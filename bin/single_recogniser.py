import cv2
import easygui
import numpy as np
from matcher import euclidean_matcher
import cv2
import numpy as np
import os
import easygui
from collections import Counter
from rgbd import rgbd
import scipy
import traceback
from matcher import euclidean_matcher
from matcher import svm_matcher_linear
from matcher import svm_matcher_poly
from matcher import svm_matcher_rbf
from matcher import rdf_matcher
from matcher import DecisionTreeMatcher
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
label=(rgbd(rgb,depth))
naming=np.load("../database/naming.npy")
locations=np.load("../database/locations.npy")

print naming[label]
#print labels[:10]

