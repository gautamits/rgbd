import os
import sys
import numpy as np
import cv2
from rgbd import rgbd
import easygui
import sklearn
#from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.externals import joblib
from sklearn import tree
#path=sys.argv[1]
"""def RandomForest(data,labels):
	forest = RandomForestClassifier(n_estimators=100, random_state=1)
	multi_target_forest = sklearn.multioutput.MultiOutputClassifier(forest, n_jobs=-1)
	multi_target_forest.fit(data, labels)
	print "random forest classifier trained"
	joblib.dump(multi_target_forest,"database/rdf.pkl")
"""

def svm_matcher(data,labels):
	svm_model=svm.SVC(kernel='linear', C=1, gamma=1) 
	svm_model.fit(data, labels)
	svm_model.score(data, labels)
	joblib.dump(svm_model,"svm.pkl")

def DecisionTree(data,labels):
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(data, labels)
	joblib.dump(clf,"DecisionTree.pkl")
try:
	data=np.load("database/data.npy")
	labels=np.load("database/labels.npy")
	print "data and labels are found. Do you want to train model directly"
	ans=raw_input("Y/N ?")
	if ans=="Y":
		print "training svm "
		svm_matcher(data,labels)
		print "svm training completed"
		print "training Decision Tree"
		DecisionTree(data,labels)
		print "decision tree training completed"
		exit()
except:
	print "need to read files"
path=easygui.diropenbox("select the root of database")
naming=[]
labels=[]
locations=[]
data=[]
k=0
for folders in os.listdir(path):  #loop over all the persons
	folder=path+'/'+folders
	naming.append(folders)
	depth_dir=folder+'/Depth'
	color_dir=folder+'/RGB'
	for images in os.listdir(color_dir):
		color_image=color_dir+'/'+images
		depth_image=depth_dir+'/'+images
		print "checking for ",color_image,
		color=cv2.imread(color_image)
		color=cv2.resize(color,(140,140))
		try:
			depth=cv2.imread(depth_image)
			depth=cv2.resize(depth,(140,140))
		except:
			print "depth image not found for this rgb image ",color_image
			continue
		try:
		
			feature=rgbd(color,depth).flatten()
			
			print "feature shape saved by database is ",feature.shape
			data.append(feature)
			#print image
			locations.append(color_image)
			labels.append(k)
		except Exception as e: 
			print e
			print "failed for ",color_image
		#data.append(hist(calcgrad(cv2.resize(cv2.imread(image,0),(240,240)))))
	k+=1
os.chdir("database")
np.save("naming",np.array(naming))
np.save("labels",np.array(labels))
np.save("locations",np.array(locations))
np.save("data",np.array(data))
#RandomForest(data,labels)
svm_matcher(data,labels)
DecisionTree(data,labels)