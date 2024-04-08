from ultralytics import YOLO
import cv2
import math

# import the model
model = YOLO(r"ASL-model4.pt")

# classes for ASL detection
classNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
              "K", "L", "M", "N", "O", "P", "Q", "R", "S", "SPACE", "T",
              "U", "V", "W", "X", "Y", "Z", "DEL"]

# start the webcam
cap = cv2.VideoCapture(0)
window_name = "Webcam"     

if not (cap.isOpened()):
    print("Could not open video device")

cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    results = model(img, stream = True)

    # coordination
    for r in results:
        boxes = r.boxes

        for box in boxes:
            if(box.conf[0]>=0.60):
                # make bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (181, 245, 10 ), 2)

                #conf
                conf = math.ceil((box.conf[0]*100))/100
                print("Confidence: ", conf)

                # class name 
                cls = int(box.cls[0])
                print("Class: ", classNames[cls])

                # object details
                org = [x1, y1 + 30]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                opt = classNames[cls] + ", " + str(conf)

                cv2.putText(img, opt, org, font, fontScale, color, thickness)
            
       
    cv2.imshow(window_name, img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
