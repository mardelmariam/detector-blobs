# -*- coding: utf-8 -*-
"""!@brief Detección de blobs en imagen difusa

@author: Maryam del Mar Correa
@version 1.0
@date 2021

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
    @param[in] val1 Umbral inferior
    @param[in] val2 Umbral superior
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

img_kernel = generate_gaussian_window(12, 96, 255, 4, 'light')
img_process1 = cv2.morphologyEx(img_sample, cv2.MORPH_OPEN, img_kernel)

DARK_SPOT_THRESHOLD = 96 
thresh_img = simple_threshold(img_process1, DARK_SPOT_THRESHOLD, 'black')

contours = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
img_result = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), dtype='uint8')
    
for i in range (0, img_result.shape[0]):
    for j in range (0, img_result.shape[1]):
        img_result[i,j,:] = img_sample[i,j]
    
for c in contours[0]:
    a = cv2.contourArea(c) 
    p = cv2.arcLength(c,True) 
    ci = p**2/(4*np.pi*a) if a!=0 else 0
    if ci > 1.1: # Evaluar circularidad
        x, y, w, h = cv2.boundingRect(c)
        if w>11 and h>11 and w<46 and h<60:
            cv2.rectangle(img_result, (x, y), (x+w, y+h), (255, 0, 0, 0), 2)
    else:
        pass
    
#%% - Detección de puntos claros
    
img_process2 = 255 - img_process1

black_kernel = generate_gaussian_window(20, 150, 255, 7, 'dark')
img_kernel1 = cv2.filter2D(1-img_sample/255, -1, kernel=black_kernel/255, borderType=cv2.BORDER_REPLICATE)
    
thresh_img = simple_threshold(img_kernel1, 96, 'white')
    
contours = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
for c in contours[0]:
    a = cv2.contourArea(c) 
    p = cv2.arcLength(c,True) 
    ci = p**2/(4*np.pi*a) if a!=0 else 0
    if ci > 1.1: # Evaluar circularidad
        x, y, w, h = cv2.boundingRect(c)
        if w>11 and h>11 and w<80 and h<80:
            cv2.rectangle(img_result, (x, y), (x+w, y+h), (255, 0, 0, 0), 2)
    else:
        pass
    
plt.figure()
plt.gcf().set_dpi(1200)
plt.subplot(1,3,1)
plt.imshow(255-img_sample, cmap='gray', vmin=0, vmax=255)
plt.subplot(1,3,2)
plt.imshow(thresh_img, cmap='gray', vmin=0, vmax=255)
plt.subplot(1,3,3)
plt.imshow(img_result, cmap='gray', vmin=0, vmax=255)
