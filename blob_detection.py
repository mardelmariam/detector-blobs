# -*- coding: utf-8 -*-
"""!@brief Detección de blobs oscuros en imagen difusa

@author: Maryam del Mar Correa
@version 1.1
@date 2022

"""

import matplotlib.pyplot as plt
import numpy as np

import os
from tifffile import TiffFile

import cv2 
import scipy
from scipy import signal 


#%%

def simple_threshold(arr, val, mode):
        
    """!@brief Genera una nueva imagen, en la que todos los píxeles por encima
    de un umbral se conservan. La imagen debe estar en escala de grises

    @param[in] arr Arreglo correspondiente a la imgen a procesar
    @param[in] val Umbral inferior
    @parain[in] mode Fondo: 'black' o 'white'
    @return thresh_img Imagen generada
    """
    
    thresh_img = np.zeros(arr.shape, dtype='uint8')
    
    if mode=='black':
        for i in range (0, arr.shape[0]):
            for j in range (0, arr.shape[1]):
                thresh_img[i,j] = 255 if arr[i,j] >= val else 0
    elif mode=='white':
        for i in range (0, arr.shape[0]):
            for j in range (0, arr.shape[1]):
                thresh_img[i,j] = 0 if arr[i,j] >= val else 255
                
    thresh_img = thresh_img.astype('uint8')
    
    return thresh_img

#%%

def generate_gaussian_window(k_size, a, b, std, color):
    
    """!@brief Devuelve una matriz gaussiana bidimensional para filtrado

    @param[in] k_size Tamaño del kernel cuadrado en píxeles
    @param[in] a Valor mínimo de escala de grises
    @parain[in] b Valor máximo de escala de grises
    @parain[in] std Desviación estándar de la función gaussiana asociada
    @parain[in] color Color central: 'dark' para oscuro y 'light' para claro
    @return gkern2d Matriz gaussiana generada
    """
    
    gkern1d = scipy.signal.windows.gaussian(k_size, std)
    gkern2d = np.outer(gkern1d, gkern1d)
    if color=='dark':
        gkern2d = (b-a)*gkern2d + a
    else:
        gkern2d = (a-b)*gkern2d + b

    return gkern2d

    
#%%

def adjust_brightness_contrast(img_arr, alpha, beta):
    
    """!@brief Realce de contraste y brillo de la imagen
    
    @param[in] img_arr Imagen a modificar
    @param[in] alpha Factor de contraste
    @parain[in] beta Factor de brillo
    @return arr Imagen modificada

    """
    arr = (img_arr + beta)*alpha
    return arr


#%% - Cargar imagen general a procesar

img_path = "ndvi_sample.tif"

with TiffFile(img_path) as tif:
    img = tif.asarray()

for i in range (0, img.shape[0]):
    for j in range (0, img.shape[1]):
        img[i,j] = 0 if img[i,j] < 0 else img[i,j]
        
img_sample = np.zeros(img.shape, dtype='uint8')

for i in range (0, img.shape[0]):
    for j in range (0, img.shape[1]):
        img_sample[i,j] = img[i,j] * 255
        
plt.figure()
plt.gcf().set_dpi(1200)
plt.imshow(img_sample, cmap='gray', vmin=0, vmax=255)

        
#%% - Detección de puntos oscuros

img_kernel = generate_gaussian_window(9, 40, 245, 1.3, 'light')
        
img_process1 = cv2.filter2D(img_sample/255.0, -1, img_kernel/255.0)
    
thresh_img = simple_threshold(img_process1, 31, 'black')
   
contours = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
img_result = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), dtype='uint8')
            
for i in range (0, img_result.shape[0]):
    for j in range (0, img_result.shape[1]):
        img_result[i,j,:] = img_sample[i,j]
            
for c in contours[0]:
    a = cv2.contourArea(c) 
    p = cv2.arcLength(c,True) 
    ci = p**2/(4*np.pi*a) if a!=0 else 0
    if ci > 0.7: # Evaluar circularidad
        x, y, w, h = cv2.boundingRect(c)
        if w>11 and h>11 and w<49 and h<49:
            cv2.rectangle(img_result, (x, y), (x+w, y+h), (255, 0, 0, 0), 2)
        else:
            pass
        
fig = plt.figure()
fig.suptitle("Detección de puntos oscuros", x=0.5, y=0.75)
plt.gcf().set_dpi(1200)
plt.subplot(1,3,1)
plt.imshow(img_process1, cmap='gray', vmin=0, vmax=255)
plt.subplot(1,3,2)
plt.imshow(thresh_img, cmap='gray', vmin=0, vmax=255)
plt.subplot(1,3,3)
plt.imshow(img_result, cmap='gray', vmin=0, vmax=255)
    