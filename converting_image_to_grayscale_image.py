import cv2

img = cv2.imread("sample1.jpg")
cv2.imshow("sample1",img)
cv2.imwrite("sample1Copy.jpg",img)

grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("Orig",img)
cv2.imshow("Gray",grayImg)

print (img.shape) #(342, 548, 3)
print (img.size)  #562248
print (img.dtype) #uint8 


cv2.waitKey(0)
cv2.destroyAllWindows()
