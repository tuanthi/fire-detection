import cv2
import numpy as np
import os

video_file = "field.mp4"
image_dir = "./images"
ROI_x = 250
ROI_xmax = 280
ROI_y = 80
ROI_ymax = 150
width = 638
height = 360

video = cv2.VideoCapture(video_file)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_file.split('.')[0] + '.avi',fourcc, 20.0, (width,height))
i=1
while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break
    i += 1
    print(i)
    if i<965:
        continue

    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = [5, 50, 50]
    upper = [255, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)

    mask[:,:ROI_x] = np.zeros((height, ROI_x), np.int8);
    mask[:,ROI_xmax:] = np.zeros((height, width-ROI_xmax), np.int8);
    mask[:ROI_y,:] = np.zeros((ROI_y, width), np.int8);
    mask[ROI_ymax:,:] = np.zeros((height-ROI_ymax, width), np.int8);

    cv2.imshow("omask", mask)

    kernel1 = np.ones((3,3),np.uint8)
    kernel2 = np.ones((10,10),np.uint8)
    mask = cv2.erode(mask,kernel1,iterations = 1)
    mask = cv2.dilate(mask,kernel2,iterations = 5)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    output = cv2.bitwise_and(frame, hsv, mask=mask)

    cnt = cv2.findNonZero(mask)
    if cnt is not None:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        fire = cv2.drawContours(frame, [box], 0, (0,0,255), 2)
        cv2.putText(frame,'Fire Detected!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow("Fire Detection", fire)

    cv2.imshow("mask", mask)
    cv2.imshow("output", output)
    cv2.imshow("original", frame)
    out.write(frame)

    if i > 1730:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()
out.release()


# # Tuning color
# video_file = "garage.mp4"
# lower = [175, 50, 50]
# upper = [180, 255, 255]

# chef, fire
# lower = [27, 50, 50]
# upper = [35, 255, 255]