import cv2
from rgbd import rgbd
import numpy as np
import scipy
from sklearn import svm
from sklearn.externals import joblib
from collections import Counter
data=np.load("database/data.npy")
labels=np.load("database/labels.npy")
locations=np.load("database/locations.npy")
try:
	svm_model=svm.SVC(kernel='linear', C=1, gamma=1) 
	svm_model=joblib.load("database/svm.pkl")
except:
	svm_model=svm.SVC(kernel='linear', C=1, gamma=1) 
	svm_model.fit(data, labels)
	svm_model.score(data, labels)
	joblib.dump(svm_model,"database/svm.pkl")
def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]
def euclidean_matcher(feature):
	distance=[]
	k=0
	for i in data:
		dist=scipy.spatial.distance.euclidean(i,feature)
		#print locations[k],labels[k],dist
		distance.append(dist)
		k+=1
	temp=labels.copy()
	distance,temp=zip(*sorted(zip(distance,temp)))
	if distance[0]==0:
		temp=temp[1:11]
	else:
		temp=temp[:10]
	label=Most_Common(temp)
	return label
def svm(feature):
	
	"""model = svm.svc(kernel='linear', c=1, gamma=1) 
	# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
	model.fit(data, labels)
	model.score(data, labels)
	#Predict Output
	"""
	predicted= svm_model.predict(feature)
	return predicted