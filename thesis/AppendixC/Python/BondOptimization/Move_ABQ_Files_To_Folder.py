# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:02:44 2020

@author: Kiffer Creveling
"""
      
# importing os module  
import os
import glob
import shutil

def MoveAbqFiles(fileName, folderDirectory, abqWD):
    
    # """ Change directory to correct path """
    
    # dataDirectory = os.path.join(abqWD, fileName)
    # if not os.path.exists(dataDirectory):
    #     os.makedirs(dataDirectory)
    
    # List of files in the ABQ working directory with the same name as the 
    # 'fileName''
    fileList = glob.glob('{}.*'.format(os.path.join(abqWD, fileName)))
    for i in fileList:
        if i == folderDirectory:
            # Skip the file with the exact same name (i.e. Folder name...)
            continue
        source = os.path.join(abqWD,i)
        destination = os.path.join(folderDirectory)
        # copy (since shutil.move wouldn't overwrite)
        dest = shutil.copy(source, destination)
        os.remove(source) # remove the source file
        
    return print('Files moved = :)')
