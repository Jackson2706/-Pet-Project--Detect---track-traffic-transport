import torchvision
import torch
import argparse
import cv2
import detect_utils
from PIL import Image
import time

# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the model
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
# load the model on to the computation device
model.eval().to(device)

vid = cv2.VideoCapture("rtsp://admin:Admin@123@27.72.149.50:1554/profile3/media.smp")
count = 0
while True:
    ret, frame = vid.read()
    count +=1
    t = time.time()
    
    # frame = cv2.resize(frame, (300,300))
    image = Image.fromarray(frame)

    # detect outputs
    boxes, classes, labels = detect_utils.predict(image, model, device, 0.5)
    # draw bounding boxes
    image = detect_utils.draw_boxes(boxes, classes, labels, image)
    cv2.imshow('Image', image)
    duration = time.time() - t
    print(1/duration)

    torch.cuda.empty_cache()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()  
                                                                                  
# cv2.imwrite(f"outputs/{save_name}.jpg", image)
# cv2.waitKey(0)