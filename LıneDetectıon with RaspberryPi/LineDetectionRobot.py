import cv2 as cv
import numpy as np
import time
import argparse

def do_canny(frame):
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 50, 150)
    return canny

def do_segment(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    polygons = np.array([
                            [(0, height), (800, height), (380, 290)]
                        ])
    mask = np.zeros_like(frame)
    cv.fillPoly(mask, polygons, 255)
    segment = cv.bitwise_and(frame, mask)
    return segment

def calculate_lines(frame, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    y1 = frame.shape[0]
    y2 = int(y1 - 150)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    lines_visualize = np.zeros_like(frame)
    
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            ppt3 = np.array([
                lines[1][0],lines[1][1],
                lines[1][2],lines[1][3],
                lines[1][2],lines[1][3]-100,
                lines[1][0],lines[1][1]-150
                ],np.int32)
            ppt3 = ppt3.reshape((-1,1,2))
            cv.fillPoly(lines_visualize,[ppt3],(0,255,0),32)

            ppt2 = np.array([
                lines[0][0],lines[0][1],
                lines[0][2],lines[0][3],
                lines[0][2],lines[0][3]-100,
                lines[0][0],lines[0][1]-150
                ],np.int32)
            ppt2 = ppt2.reshape((-1,1,2))
            cv.fillPoly(lines_visualize,[ppt2],(0,255,0),32)
            
            
            ppt = np.array([
                lines[0][0],lines[0][1],
                lines[1][0],lines[1][1],
                lines[1][2],lines[1][3],
                lines[0][2],lines[0][3]
                ],np.int32)
            ppt = ppt.reshape((-1,1,2))
            cv.fillPoly(lines_visualize,[ppt],(0,0,255),32)
            
            
            x1= int(((lines[1][0]-lines[0][0])/2)) + lines[0][0]
            y1= int(((lines[1][1]-lines[0][1])/2)) + lines[0][1]
            x2= int(((lines[1][2]-lines[0][2])/2)) + lines[0][2]
            y2= int(((lines[1][3]-lines[0][3])/2)) + lines[0][3]
            cv.line(lines_visualize, (x1,y1), (x2,y2), (255, 255, 255), 2)
            cv.line(lines_visualize, (int((x1+x2)/2),int((y1+y2)/2-50)), (int((x1+x2)/2),int((y1+y2)/2)+50), (0, 255, 255), 2)
            
            
            komsu = x1
            karsi = y1
            print("ACI-------",np.arcsin((komsu)/(karsi)))     
    return lines_visualize

cap = cv.VideoCapture("input.mp4")
cap2 = cv.VideoCapture("input.mp4")

while (cap.isOpened()):
    ret, frame = cap.read()
    
    canny = do_canny(frame)
    
    segment = do_segment(canny)
    
    hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 10, maxLineGap = 50)
    
    lines = calculate_lines(frame, hough)
    
    merkezx = ((int(((lines[1][0]-lines[0][0])/2)) + lines[0][0])+(int(((lines[1][2]-lines[0][2])/2)) + lines[0][2]))/2
    solx = (lines[0][0] + lines[0][2]) / 2
    sagx = (lines[1][0] + lines[1][2]) / 2
    print("sol-----",np.abs(merkezx - solx))
    print("sag----",np.abs(merkezx - sagx))
    
    lines_visualize = visualize_lines(frame, lines)

    output = cv.addWeighted(frame, 0.9, lines_visualize, 1, 1)
    
    cv2.imshow("LineDetection", output)
   
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()