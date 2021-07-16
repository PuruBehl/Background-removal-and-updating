from background.functions import grabcut_function,fill_image
import matplotlib.pyplot as plt
import cv2
import numpy as np

image_path = "images\image3.jpeg"
background_image_path = r"images/background2.jpeg"
size = (1800,1000)

image=cv2.imread(image_path)
background_image = cv2.imread(background_image_path)

cv2.imshow("image",background_image)

image_resized = cv2.resize(image,size)

new_img = grabcut_function(image_resized,500,10,"gaussian_blur")

fill_image(new_img,background_image)

cv2.waitKey(0)