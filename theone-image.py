import cv2
import numpy as np

# Load Yolo
print("LOADING TENSORFLOW")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# nessary
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
#get
layer_names = net.getLayerNames()

#Determine the output layer names from the YOLO model
outputlayers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
print("TENSORFLOW LOADED")

# Capture image

#img = cv2.imread("1.jpg")
#img = cv2.imread("2.jpg")
#img = cv2.imread("3.jpg")
#img = cv2.imread("4.jpg")
#img = cv2.imread("5.jpg")
#img = cv2.imread("6.jpg")
#img = cv2.imread("7.jpg")
#img = cv2.imread("8.jpg")
#img = cv2.imread("9.jpg")
#img = cv2.imread("10.jpg")
img = cv2.imread("11.jpg")
#img = cv2.imread("12.jpg")
#img = cv2.imread("13.jpg")
#img = cv2.imread("14.jpg")
#img = cv2.imread("15.jpg")
#img = cv2.imread("16.jpg")
#img = cv2.imread("17.jpg")
#img = cv2.imread("18.jpg")
#img = cv2.imread("19.jpg")
#img = cv2.imread("20.jpg")
#img = cv2.imread("21.jpg")
#img = cv2.imread("22.jpg")

#img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# USing blob function of opencv to preprocess image
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)
# Detecting objects
net.setInput(blob)
outs = net.forward(outputlayers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs :
    for detection in out :
        scores = detection[5 :]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 :
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# We use NMS function in opencv to perform Non-maximum Suppression
# we give it score threshold and nms threshold as arguments.
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in range(len(boxes)) :
    if i in indexes :
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    1 / 2, color, 2)

cv2.imshow("Image", img)
cv2.waitKey(0)