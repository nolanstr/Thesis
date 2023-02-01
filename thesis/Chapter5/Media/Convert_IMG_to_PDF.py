# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:16:03 2021

@author: Kiffer Creveling
"""
import os
import numpy as np
import glob
from PIL import Image

# In[file list]

wd = os.getcwd()

# filter images
fileListJPG = glob.glob('{}//*.JPG'.format(wd))
fileListPNG = glob.glob('{}//*.PNG'.format(wd))
fileListjpg = glob.glob('{}//*.jpg'.format(wd))
fileListpng = glob.glob('{}//*.png'.format(wd))

# append all images to list to be converted
fileList = []
fileList.append(fileListJPG)
fileList.append(fileListPNG)
fileList.append(fileListjpg)
fileList.append(fileListpng)

listofLists = [item for sublist in fileList for item in sublist]

# In[Convert images to pdf]

for i in listofLists:
    image1 = Image.open(i)
    im1 = image1.convert('RGB')
    
    # Split the file extension
    imName, ext = os.path.splitext(i)
    
    im1.save(imName + '.pdf')

