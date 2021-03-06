import cv2
import fast
import ORB

cap = cv2.VideoCapture('../input/p2-1-0.mp4')
cont = 1 
while(cap.isOpened()):
		print('=== FRAME : {0:2d} ==='.format(cont))
		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		keyPoints = fast.interest_points(gray,threshold=30,N=8)
		keyPoints = ORB.harris_measure_and_orientation(gray, keyPoints, 300)
		cv2.imshow("Interest Point",cv2.drawKeypoints(frame, keyPoints, color=(0,255,0), flags=0))
		cont = cont + 1
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()
