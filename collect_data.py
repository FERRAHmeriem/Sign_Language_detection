import os
import cv2

DATA_DIR = './collected_data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR);

nb_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(0);

for i in range(nb_classes) :
    if not os.path.exists(os.path.join(DATA_DIR , str(i))):
        os.makedirs(os.path.join(DATA_DIR , str(i)))
    
    #collecting data :
    while True : 
        ret , frame = cap.read()
        cv2.putText(frame , 'READY! Press "Q" :' , (100 , 50) , cv2.FONT_HERSHEY_SIMPLEX , 1.3 , (0 , 244 , 0) , 3 , cv2.LINE_AA)
        cv2.imshow( 'frame' ,frame)
        if cv2.waitKey(30) == ord('q') :
            break
    
    counter = 0 ;
    while counter < dataset_size :
        ret , frame = cap.read()
        cv2.imshow('frame' , frame)
        cv2.waitKey(30)
        cv2.imwrite(os.path.join(DATA_DIR , str(i) , '{}.jpg'.format(counter)) , frame)
        counter +=  1

cap.release()
cv2.destroyAllWindows()