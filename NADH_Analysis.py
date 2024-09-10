import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim

VIDEO="NADH_Speed.m4v"
LOWER = np.array([0,0,100])
UPPER = np.array([70,70,250])
GRAPH=False
SAVING=True
DATA=[]


def apply_mask(frame, _mode='simple'):
	if _mode=='simple':
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mask = cv2.inRange(image, LOWER, UPPER)
		
	elif _mode=='otsu':
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		image = cv2.GaussianBlur(gray, (7, 7), 0)
		ret, mask = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	result = cv2.bitwise_and(frame, frame, mask=mask)

	return image, mask, result


def process_frame(frame):
	image, mask, result=apply_mask(frame)

	# Append sum of pixel masked to general DATA list
	px_int=result.sum()
	DATA.append(px_int)

	return image, mask, result


def update_graph(i):

	rete, frame=capture.read()
	if not rete: return
	else:
		image, mask, result = process_frame(frame)

		ax0.clear()
		ax0.imshow(result, cmap='winter')

		ax1.clear()
		ax1.plot(DATA,c='r')
		#ax1.set_ylim(0, 30000000)
		ax1.set_xlim(0,length)



capture=cv2.VideoCapture(VIDEO)
length=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

if SAVING:
	# Do the background processing
	while capture.isOpened():
		rete, frame=capture.read()
		if not rete: break
		else:process_frame(frame)

	df=pd.Series(DATA)
	df=df.loc[df>30000000].ewm(span=20).mean()
	df.to_csv(f'{VIDEO}_analysis.csv')
	plt.plot(df)
	plt.show()



if GRAPH:

	DATA=[]
	# Initiate graph that will plot pixel count
	fig, (ax0, ax1) =plt.subplots(2,1)
	#line, =ax1.plot([], [], c='r')

	while capture.isOpened():
		
		# do the graph after...
		ani=anim.FuncAnimation(fig, update_graph, frames=length, interval=20)
		plt.winter()
		plt.show()


	# Close the window
	capture.release()

	# De-allocate any associated memory usage
	cv2.destroyAllWindows()





