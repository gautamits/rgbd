import cv2
def histogram(image):
	#image = cv2.imread("test.jpg",0)
	winSize = (70,70)
	blockSize = (70,70)
	blockStride = (5,5)
	cellSize = (35,35)
	nbins = 64
	derivAperture = 1
	winSigma = 4.
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
	                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
	#compute(img[, winStride[, padding[, locations]]]) -> descriptors
	winStride = (8,8)
	padding = (8,8)
	locations = ((10,20),)
	#hist = hog.compute(image,winStride,padding,locations)
	hist=hog.compute(image)
	#print "histogram shape found by hog is ",hist.shape
	return hist
def histogram2(image,horizontal_slices,vertical_slices):
	#shape function returns height first and then width
	h,w=image.shape[:2]
	#print image.shape
	#print h_slices,w_slices
	if w%horizontal_slices!=0:
		#cannot be sliced into equal sizes
		n=w/horizontal_slices
		w=horizontal_slices*(n+1)
	if h%vertical_slices!=0:
		#cannot be sliced into equal sizes
		n=h/vertical_slices
		h=vertical_slices*(n+1)
	image=cv2.resize(image,(w,h))
	print image.shape

