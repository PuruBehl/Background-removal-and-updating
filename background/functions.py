import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt



def grabcut_function(image,size,iterations=5,smoothing="median_blur"):
    ''' this function takes in an image and removes the background using the grabcut feature of open cv '''
    img=image
    img=cv2.resize(img,(size,size))
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (50,50,400,350)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,iterations,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    img = cv2.resize(img,(1800,1000))
    img = remove_noise(img,smoothing)
    plt.imshow(img),plt.colorbar(),plt.show()
    return img

def remove_noise(image,options="median_blur") :
    ''' This function is to remove noise from the image '''
    if options=="median_blur":
        median_blur_image=cv2.medianBlur(image,3)
        return median_blur_image
    elif options == "gaussian_blur" :
        gaussian_blur_image = cv2.GaussianBlur(image, (7, 7), 0)
        return gaussian_blur_image 



def fill_image(image,background_image) :
    ''' This function is to fill the image with a new background image'''
  
    background_image = cv2.resize(background_image,(image.shape[1],image.shape[0]))

    new_image = []
    print(image.shape)
    print(background_image.shape)
    for i in range(image.shape[0]) :
        main_image=[]
        for j in range(image.shape[1]) :
            if list(image[i][j]) == [0,0,0] :
                main_image.append(background_image[i][j])
            else :
                main_image.append(image[i][j])
        new_image.append(main_image)

    new_image = np.array(new_image)

    print(new_image.shape)

    plt.imshow(new_image),plt.colorbar(),plt.show()

