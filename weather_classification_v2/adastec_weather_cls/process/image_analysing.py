import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%
img_sun = cv2.imread("/home/enesdrk/Desktop/70255.jpg")
img_cloud = cv2.imread("/home/enesdrk/Desktop/86212.jpg")
img_rain = cv2.imread("/home/enesdrk/Desktop/88754.jpg")


img_sun_gray = cv2.cvtColor(img_sun, cv2.COLOR_BGR2GRAY)
img_cloud_gray = cv2.cvtColor(img_cloud, cv2.COLOR_BGR2GRAY)
img_rain_gray = cv2.cvtColor(img_rain, cv2.COLOR_BGR2GRAY)

#%%
def calc_hga(image):
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    
    edges_x = cv2.filter2D(image,cv2.CV_8U,kernelx)
    edges_y = cv2.filter2D(image,cv2.CV_8U,kernely)
    
        
    features = np.zeros([edges_x.shape[0], edges_x.shape[1]])
    features_flatten = []
    
    for j in range(edges_x.shape[0]):
        for k in range(edges_x.shape[1]):
            features[j,k] = int(np.sqrt(np.power(edges_x[j,k], 2) + np.power(edges_y[j,k], 2)))
            features_flatten.append(features[j,k])
    
    return features, features_flatten

def Average(lst):
    return sum(lst) / len(lst)

#%%

sun_hsv = cv2.cvtColor(img_sun, cv2.COLOR_BGR2HSV)
rain_hsv = cv2.cvtColor(img_rain, cv2.COLOR_BGR2HSV)
cloud_hsv = cv2.cvtColor(img_cloud, cv2.COLOR_BGR2HSV)

h_sun, s_sun, v_sun = sun_hsv[:,:,0], sun_hsv[:,:,1], sun_hsv[:,:,2]
h_cloud, s_cloud, v_cloud = cloud_hsv[:,:,0], cloud_hsv[:,:,1], cloud_hsv[:,:,2]
h_rain, s_rain, v_rain = rain_hsv[:,:,0], rain_hsv[:,:,1], rain_hsv[:,:,2]

sun_hist_h = cv2.calcHist([h_sun],[0],None,[256],[0,256])
sun_hist_s = cv2.calcHist([s_sun],[0],None,[256],[0,256])
sun_hist_v = cv2.calcHist([v_sun],[0],None,[256],[0,256])

plt.plot(sun_hist_h, color='r', label="h_sun")
plt.plot(sun_hist_s, color='g', label="s_sun")
plt.plot(sun_hist_v, color='b', label="v_sun")
plt.legend()
plt.show()

cloud_hist_h = cv2.calcHist([h_cloud],[0],None,[256],[0,256])
cloud_hist_s = cv2.calcHist([s_cloud],[0],None,[256],[0,256])
cloud_hist_v = cv2.calcHist([v_cloud],[0],None,[256],[0,256])

plt.plot(cloud_hist_h, color='r', label="h_cloud")
plt.plot(cloud_hist_s, color='g', label="s_cloud")
plt.plot(cloud_hist_v, color='b', label="v_cloud")
plt.legend()
plt.show()

rain_hist_h = cv2.calcHist([h_rain],[0],None,[256],[0,256])
rain_hist_s = cv2.calcHist([s_rain],[0],None,[256],[0,256])
rain_hist_v = cv2.calcHist([v_rain],[0],None,[256],[0,256])

plt.plot(rain_hist_h, color='r', label="h_rain")
plt.plot(rain_hist_s, color='g', label="s_rain")
plt.plot(rain_hist_v, color='b', label="v_rain")
plt.legend()
plt.show()

#%%

sobelx_sun = cv2.Sobel(img_sun_gray,cv2.CV_64F,1,0,ksize=5)
sobely_sun = cv2.Sobel(img_sun_gray,cv2.CV_64F,0,1,ksize=5)

features_grd_sun = np.zeros([sobelx_sun.shape[0], sobelx_sun.shape[1]])
features_flatten_grd_sun = []

for j in range(sobelx_sun.shape[0]):
    for k in range(sobelx_sun.shape[1]):
        features_grd_sun[j,k] = int(np.sqrt(np.power(sobelx_sun[j,k], 2) + np.power(sobely_sun[j,k], 2)))
        features_flatten_grd_sun.append(features_grd_sun[j,k])



sobelx_cloud = cv2.Sobel(img_cloud_gray,cv2.CV_64F,1,0,ksize=5)
sobely_cloud = cv2.Sobel(img_cloud_gray,cv2.CV_64F,0,1,ksize=5)

features_grd_cloud = np.zeros([sobelx_cloud.shape[0], sobelx_cloud.shape[1]])
features_flatten_grd_cloud = []

for j in range(sobelx_cloud.shape[0]):
    for k in range(sobelx_cloud.shape[1]):
        features_grd_cloud[j,k] = int(np.sqrt(np.power(sobelx_cloud[j,k], 2) + np.power(sobely_cloud[j,k], 2)))
        features_flatten_grd_cloud.append(features_grd_cloud[j,k])
        

sobelx_rain = cv2.Sobel(img_rain_gray,cv2.CV_64F,1,0,ksize=5)
sobely_rain = cv2.Sobel(img_rain_gray,cv2.CV_64F,0,1,ksize=5)

features_grd_rain = np.zeros([sobelx_rain.shape[0], sobelx_rain.shape[1]])
features_flatten_grd_rain = []

for j in range(sobelx_rain.shape[0]):
    for k in range(sobelx_rain.shape[1]):
        features_grd_rain[j,k] = int(np.sqrt(np.power(sobelx_rain[j,k], 2) + np.power(sobely_rain[j,k], 2)))
        features_flatten_grd_rain.append(features_grd_rain[j,k])

print("sun max G = ", max(features_flatten_grd_sun))
print("cloud max G = ", max(features_flatten_grd_cloud))
print("rain max G = ", max(features_flatten_grd_rain))

#cloud_norm = ((features_grd_cloud - features_grd_cloud.min()) * (1/(features_grd_cloud.max() - features_grd_cloud.min()) * 255)).astype('uint8')

#cv2.imshow("grad sun", cloud_norm)
#cv2.waitKey(5000)
#cv2.destroyAllWindows()

#%%    

sun_hga_ftrs = ((features_grd_sun - features_grd_sun.min()) * (1/(features_grd_sun.max() - features_grd_sun.min()) * 255)).astype('uint8')
sun_hga_hist = cv2.calcHist([sun_hga_ftrs],[0],None,[256],[0,256])

plt.hist(sun_hga_hist, bins=[0,10,20,30,40,50,60,70,80,90,99])
plt.show()

rain_hga_ftrs = ((features_grd_rain - features_grd_rain.min()) * (1/(features_grd_rain.max() - features_grd_rain.min()) * 255)).astype('uint8')
rain_hga_hist = cv2.calcHist([rain_hga_ftrs],[0],None,[256],[0,256])

plt.hist(rain_hga_hist, bins=[0,10,20,30,40,50,60,70,80,90,99])
plt.show()

cloud_hga_ftrs = ((features_grd_cloud - features_grd_cloud.min()) * (1/(features_grd_cloud.max() - features_grd_cloud.min()) * 255)).astype('uint8')
cloud_hga_hist = cv2.calcHist([cloud_hga_ftrs],[0],None,[256],[0,256])

plt.hist(cloud_hga_hist, bins=[0,10,20,30,40,50,60,70,80,90,99])
plt.show()

#%%

img_rect = img_sun.copy()

cv2.rectangle(img_rect, (540, 790), (650, 820), (0,0,255), 2)

x = [k for k in range(540, 650, 10)]
y = [j for j in range(790, 820 ,10)]


for i in range(len(x)):
    for j in range(len(y)):
        cv2.circle(img_rect, (x[i], y[j]), 1, (0,0,255), 1)


cv2.imshow("ROI", img_rect)
cv2.waitKey(5000)
cv2.destroyAllWindows()
#%%

roi_x = [k for k in range(540, 650, 10)]
roi_y = [j for j in range(790, 820 ,10)]


roi_ort_cloud = []
img_roi_cloud = img_cloud.copy()

roi_ort_rain = []
img_roi_rain = img_rain.copy()

roi_ort_sun = []
img_roi_sun = img_sun.copy()

for i in range(len(roi_x)-1):
    for j in range(len(y)-1):
        ort_sun = int(img_roi_sun[roi_x[i]:roi_x[i+1], roi_y[j]:roi_y[j+1]].mean())
        roi_ort_sun.append(ort_sun)

for i in range(len(roi_x)-1):
    for j in range(len(roi_y)-1):
        ort_cloud = int(img_roi_cloud[roi_x[i]:roi_x[i+1], roi_y[j]:roi_y[j+1]].mean())
        roi_ort_cloud.append(ort_cloud)

for i in range(len(roi_x)-1):
    for j in range(len(roi_y)-1):
        ort_rain = int(img_roi_rain[roi_x[i]:roi_x[i+1], roi_y[j]:roi_y[j+1]].mean())
        roi_ort_rain.append(ort_rain)

print("ROI sun average = ", Average(roi_ort_sun))
print("ROI cloud average = ", Average(roi_ort_cloud))
print("ROI rain average = ", Average(roi_ort_rain))

