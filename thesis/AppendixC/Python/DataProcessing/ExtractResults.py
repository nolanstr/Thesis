# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 12:12:57 2021

@author: Kiffer2
"""

import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import pdb

cwd = os.getcwd() # Get current working directory

OP = 'Results'

# In[Elastic Modulus]
ElastModResults = os.path.join(cwd, OP, 'ElasticModulusPlots')

elasticModFileName = '*optTIE_SlopeCompare.pdf'

# Search the folder directory for the file name that matches
for path in Path(cwd).rglob(elasticModFileName):
    
    
    filePathList = os.path.normpath(path).split(os.path.sep)
    
    # Look in the "Finished" folder
    if 'Finished' in filePathList:
        
        # Data trace
        dataTrace = filePathList[4] # Specific data trace
        
        print(dataTrace, path.name)
        
        # New file name (NFN)
        NFN = dataTrace + '.pdf'
        
        # Copy search results to the destination folder
        try:
            NP = os.path.join(cwd, ElastModResults) # New path
            
            # Create folder if it doesn't exist
            os.makedirs(NP, exist_ok=True)
            
            shutil.copy(path, os.path.join(NP, NFN)) # move files
        except shutil.SameFileError:
            pass



# In[Elastic Modulus Attributes]
ElastModAttr = os.path.join(cwd, OP, 'ElasticModulusAttr')

initialName = 'output'
elasticModFileAttr = '*optTIE.txt'

# Search the folder directory for the file name that matches
for path in Path(cwd).rglob(elasticModFileAttr):
    
    
    # Search for the input parameters
    if path.name.find(initialName) < 0:
        
        
        filePathList = os.path.normpath(path).split(os.path.sep)
        
        # Look in the "Finished" folder
        if 'Finished' in filePathList:
            
            # Data trace
            dataTrace = filePathList[4] # Specific data trace
            
            print(dataTrace, path.name)
            
            # New file name (NFN)
            NFN = dataTrace + '.txt'
            
            # Copy search results to the destination folder
            try:
                NP = os.path.join(cwd, ElastModAttr) # New path
                
                # Create folder if it doesn't exist
                os.makedirs(NP, exist_ok=True)
                
                shutil.copy(path, os.path.join(NP, NFN)) # move files
            except shutil.SameFileError:
                pass


# In[Elastic Modulus Convergence]
ElastModConv = os.path.join(cwd, OP, 'ElasticModulusConvergence')

ElastModConvAttr = 'FEAAttributes.txt'

# Search the folder directory for the file name that matches
for path in Path(cwd).rglob(ElastModConvAttr):
    
    # Split file path to a list
    filePathList = os.path.normpath(path).split(os.path.sep)
    
    # Look in the "Finished" folder
    if 'Finished' in filePathList:
        
        if filePathList[6].find('optTIE') > 0:
            
            # Data trace
            dataTrace = filePathList[4] # Specific data trace
            
            print(dataTrace, path.name)
            
            # New file name (NFN)
            NFN = dataTrace + '.txt'
            
            # Load Data
            
            # Add names to file because 'SimSlope','SimMax', 'SimSS' 
            # were missing
            names = ['FileName', 'Time', 'E1', 'E2', 'PT', 'G', 'V1', 'V2', 
                     'R', 'F', 'MS', 'RE', 'VE', 'Knn', 'Kss', 'Ktt', 
                     'DamageInitiation', 'tn', 'ts', 'tt', 'DamageEvolution', 
                     'FE', 'Optimization', 'TIE', 'errorListL2Norm', 
                     'ObjectiveFunction', 'SimSlope','SimMax', 'SimSS', 
                     'simTime']
            df = pd.read_csv(path, names=names, sep='\t', header=0)
            
            # Save Dat
            df.to_csv(path, sep='\t', index=False, na_rep='nan')
            
            # Copy search results to the destination folder
            try:
                NP = os.path.join(cwd, ElastModConv) # New path
                
                # Create folder if it doesn't exist
                os.makedirs(NP, exist_ok=True)
                
                shutil.copy(path, os.path.join(NP, NFN)) # move files
                
                df.to_csv(os.path.join(NP, NFN), sep='\t', index=False, 
                      na_rep='nan')
            except shutil.SameFileError:
                pass

# In[Cohesive Behavior Plots]
CohesiveResults = os.path.join(cwd, OP, 'CohesiveBehaviorPlots')

CohesiveFileName = '*opt_SlopeCompare.pdf'

# Search the folder directory for the file name that matches
for path in Path(cwd).rglob(CohesiveFileName):
    
    filePathList = os.path.normpath(path).split(os.path.sep)
    
    # Look in the "Finished" folder
    if 'Finished' in filePathList:
        
        # Data trace
        dataTrace = filePathList[4] # Specific data trace
        
        print(dataTrace, path.name)
        
        # New file name (NFN)
        NFN = dataTrace + '.pdf'
        
        # Copy search results to the destination folder
        try:
            NP = os.path.join(cwd, CohesiveResults) # New path
                
            # Create folder if it doesn't exist
            os.makedirs(NP, exist_ok=True)
            
            shutil.copy(path, os.path.join(NP, NFN)) # move files
        except shutil.SameFileError:
            pass


# In[Cohesive Behavior Attributes]
CohesiveAttr = os.path.join(cwd, OP, 'CohesiveBehaviorAttr')

initialName = '_T3_C'
elasticModFileAttr = '*opt.txt'

# Search the folder directory for the file name that matches
for path in Path(cwd).rglob(elasticModFileAttr):
    
    # Search for the input parameters
    if path.name.find(initialName) < 0:
        
        filePathList = os.path.normpath(path).split(os.path.sep)
        
        # Look in the "Finished" folder
        if 'Finished' in filePathList:
            
            # Data trace
            dataTrace = filePathList[4] # Specific data trace
            
            print(dataTrace, path.name)
            
            # New file name (NFN)
            NFN = dataTrace + '.txt'
            
            # Copy search results to the destination folder
            try:
                NP = os.path.join(cwd, CohesiveAttr) # New path
                
                # Create folder if it doesn't exist
                os.makedirs(NP, exist_ok=True)
                
                shutil.copy(path, os.path.join(NP, NFN)) # move files
            except shutil.SameFileError:
                pass


# In[Cohesive Behavior Convergence]
CohConv = os.path.join(cwd, OP, 'CohesiveBehaviorConvergence')

CohConvAttr = 'FEAAttributes.txt'

# Search the folder directory for the file name that matches
for path in Path(cwd).rglob(CohConvAttr):
    
    # Split file path to a list
    filePathList = os.path.normpath(path).split(os.path.sep)
    
    # Look in the "Finished" folder
    if 'Finished' in filePathList:
        
        if filePathList[5].find('optTIE') == -1:
            
            # Data trace
            dataTrace = filePathList[4] # Specific data trace
            
            print(dataTrace, path.name)
            
            # New file name (NFN)
            NFN = dataTrace + '.txt'
            
            # Load Data
            
            # Add names to file because 'SimSlope','SimMax', 'SimSS' 
            # were missing
            names = ['FileName', 'Time', 'E1', 'E2', 'PT', 'G', 'V1', 'V2', 
                     'R', 'F', 'MS', 'RE', 'VE', 'Knn', 'Kss', 'Ktt', 
                     'DamageInitiation', 'tn', 'ts', 'tt', 'DamageEvolution', 
                     'FE', 'Optimization', 'TIE', 'errorListL2Norm', 
                     'ObjectiveFunction', 'SimSlope','SimMax', 'SimSS', 
                     'simTime']
            df = pd.read_csv(path, names=names, sep='\t', header=0)
            
            # Save Dat
            df.to_csv(path, sep='\t', index=False, na_rep='nan')
            
            
            
            # Copy search results to the destination folder
            try:
                NP = os.path.join(cwd, CohConv) # New path
                
                # Create folder if it doesn't exist
                os.makedirs(NP, exist_ok=True)
                
                shutil.copy(path, os.path.join(NP, NFN)) # move files
                
                df.to_csv(os.path.join(CohConv, NFN), sep='\t', index=False, 
                      na_rep='nan')
            except shutil.SameFileError:
                pass

# In[Fracture Energy Integral]

FEInt = os.path.join(cwd, OP, 'FractureEnergyIntegrals')

elasticModFileAttr = 'GcSelection.pdf'

# Search the folder directory for the file name that matches
for path in Path(cwd).rglob(elasticModFileAttr):
    
    filePathList = os.path.normpath(path).split(os.path.sep)
    
    # Look in the "Finished" folder
    if 'Finished' in filePathList:
        
        # Data trace
        dataTrace = filePathList[4] # Specific data trace
        
        print(dataTrace, path.name)
        
        # New file name (NFN)
        NFN = dataTrace + '.pdf'
        
        # Copy search results to the destination folder
        try:
            NP = os.path.join(cwd, FEInt) # New path
            
            # Create folder if it doesn't exist
            os.makedirs(NP, exist_ok=True)
            
            shutil.copy(path, os.path.join(NP, NFN)) # move files
        except shutil.SameFileError:
            pass


# In[YouTube video links]

YouTube = os.path.join(cwd, OP, 'YouTube')

YouTubeFile = 'YouTubeLink.txt'

# Search the folder directory for the file name that matches
for path in Path(cwd).rglob(YouTubeFile):
    
    filePathList = os.path.normpath(path).split(os.path.sep)
    
    # Look in the "Finished" folder
    if 'Finished' in filePathList:
        
        # Data trace
        dataTrace = filePathList[4] # Specific data trace
        
        print(dataTrace, path.name)
        
        # New file name (NFN)
        NFN = dataTrace + '.txt'
        
        # Copy search results to the destination folder
        try:
            NP = os.path.join(cwd, YouTube) # New path
            
            # Create folder if it doesn't exist
            os.makedirs(NP, exist_ok=True)
            
            shutil.copy(path, os.path.join(NP, NFN)) # move files
        except shutil.SameFileError:
            pass