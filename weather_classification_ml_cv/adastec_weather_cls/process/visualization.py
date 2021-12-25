import os
import cv2
import numpy as np

'''
def plotting(img, symbol, output = 'Sunny', th_min = 230, th_max = 255,
            b = 51, g = 153, r = 255, orgx = 260, orgy = 180):

def plotting(img, symbol, output = 'Cloudy', th_min = 230, th_max = 255,
            b = 32, g = 32, r = 32, orgx = 260, orgy = 190):

def plotting(img, symbol, output = 'Rainy', th_min = 230, th_max = 255,
            b = 255, g = 229, r = 204, orgx = 270, orgy = 180):
'''

def plotting(img, symbol, output = 'Cloudy', th_min = 230, th_max = 255,
            b = 32, g = 32, r = 32, orgx = 260, orgy = 190,
            rain = 0, sun = 0, cloud = 0):

    rows,cols,channels = symbol.shape
    roi = img[200:rows + 200, 240:cols + 240]

    symbol_gray = cv2.cvtColor(symbol, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(symbol_gray, th_min, th_max, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask = mask)
    img2_fg = cv2.bitwise_and(symbol, symbol, mask = mask_inv)

    dst = cv2.add(img1_bg,img2_fg)
    img[200:rows + 200, 240:cols + 240] = dst


    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.3
    org = (orgx, orgy)
    thickness = 2

    cv2.putText(img, '{}'.format(output), org, font, 
                fontScale, (b,g,r), thickness, cv2.LINE_AA)


    adastec = cv2.imread(os.path.expanduser('~') + "/Desktop/weather_classification/adastec_weather_cls/dataset/weather_symbols/adastec.png")
    adastec = cv2.resize(adastec, (150, 50))

    rows_ada,cols_ada,channels_ada = adastec.shape
    roi_ada = img[10:rows_ada + 10, 10:cols_ada + 10]

    ada_gray = cv2.cvtColor(adastec, cv2.COLOR_BGR2GRAY)

    ret_ada, mask_ada = cv2.threshold(ada_gray, th_min, th_max, cv2.THRESH_BINARY)
    mask_inv_ad = cv2.bitwise_not(mask_ada)

    ada_bg = cv2.bitwise_and(roi_ada, roi_ada, mask = mask_ada)
    ada_fg = cv2.bitwise_and(adastec, adastec, mask = mask_inv_ad)

    dst_ada = cv2.add(ada_bg,ada_fg)
    img[10:rows_ada + 10, 10:cols_ada + 10] = dst_ada


    cv2.putText(img, 'Sunny: {}'.format(sun), (10, 80), font, 
                0.5, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(img, 'Rainy: {}'.format(rain), (10,105), font, 
                0.5, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(img, 'Cloudy: {}'.format(cloud), (10, 130), font, 
                0.5, (0,0,255), 1, cv2.LINE_AA)
    
    #return img 

    

if __name__ == '__main__':
    img = cv2.imread("/home/enesdrk/Desktop/adastec_weatherReg/dataset/small/Cloudy/cloudy1.jpg")
    img = cv2.resize(img, (640, 480))

    symbol = cv2.imread("/home/enesdrk/Desktop/adastec_weatherReg/dataset/weather_symbols/cloud.png")
    symbol = cv2.resize(symbol, (160, 160))

    
    output = plotting(img, symbol)
    

    cv2.imshow("ADASTEC Adverse Weather Classification", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    