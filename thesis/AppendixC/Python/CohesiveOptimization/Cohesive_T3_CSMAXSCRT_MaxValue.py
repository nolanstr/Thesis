# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 01:06:55 2021

@author: Kiffer Creveling

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
plt.rcParams['figure.figsize'] = [16, 9]

def MaxCohesiveCSMAXSCRT_Output(fileName, dataDirectory, maxForceTime, 
                                dataCompare):
    
    """ Read in the csv file """
    dfValsn = pd.read_csv(os.path.join(dataCompare), sep="\t", nrows=29, 
                          header=None, names=['Var', 'Attribute'])
    
    """ File Attributes """
    HID =            dfValsn['Attribute'][0]
    HAGE =           dfValsn['Attribute'][1]
    HG =             dfValsn['Attribute'][2]
    HLR =            dfValsn['Attribute'][3]
    HR =             dfValsn['Attribute'][4]
    HSSi =     float(dfValsn['Attribute'][12])
    HSSf =     float(dfValsn['Attribute'][13])
    HTMax =    float(dfValsn['Attribute'][14])
    HDispMax = float(dfValsn['Attribute'][15])
    HFMax =    float(dfValsn['Attribute'][16]) # (mN)
    HFSS =     float(dfValsn['Attribute'][17])
    # (mN/m) slope from 20 seconds prior to max force value
    HSlope20 = float(dfValsn['Attribute'][20])
    
    dfn = pd.read_csv(os.path.join(dataCompare), sep="\t", header=30)
    dfn.columns = ['Time', 'Extension', 'Force']
    dfn_time = dfn.Time
    dfn_extension = dfn.Extension # mm
    dfn_force = dfn.Force*1e3 # N ---> mN
    
    """ CohesiveCSMAXSCRT Output Data """
    df = pd.read_csv(os.path.join(dataDirectory, fileName), sep="\t", header=0)
    
    t = df['Time (s)']
    
    """ The incoming data has both the Retina and Vitreous nodes associated 
    with it.  We need to split them apart and create plots for each data set 
    separately """
    
    # Filter data by the "name" of the node that begins with 1 i.e. '1000002'
    # and create a new dataframe
    dfR = df.loc[:, df.columns.str.startswith('4')] # Retina
    dfV = df.loc[:, df.columns.str.startswith('5')] # Vitreous
    
    # Turns out this is unnecessary as the time value interferes with the max
    # # Add time to dfR & dfV
    # dfR.insert(loc=0, column='Time', value=t)
    # dfV.insert(loc=0, column='Time', value=t)
    
    # Max value at specific time
    specificTime = maxForceTime
    
    # Value in the data frame that is closest to the specified time
    actualTime = min(t, key=lambda x:abs(x - specificTime))
    
    index = t[t == actualTime].index.values[0] # index
    
    dfRSelect = dfR[t < actualTime] # Selection of the data frame
    dfVSelect = dfV[t < actualTime] # Selection of the data frameSelect
    
    # determine max value in the dataframe
    retinaMaxUCRT = dfRSelect.max().max()
    vitreousMaxUCRT = dfVSelect.max().max()
    
    # return values
    return retinaMaxUCRT, vitreousMaxUCRT


def CSMAXSCRTAbqData(fileName, dataDirectory, maxForceTime, dataCompare):
    
    # """ Change directory to correct path """
    # filePath = os.getcwd()
    # data_directory = os.path.join(filePath,jobName)
    # figures_directory = os.path.join(filePath,jobName,'Figures')
    # if not os.path.exists(figures_directory):
    #     os.makedirs(figures_directory)
        
    """ Call both functions to plot Field/History data """
    global maxCohesiveCSMUCRT

    CSMAXSCRT_files = [f for f in os.listdir(dataDirectory) 
                       if os.path.isfile(os.path.join(dataDirectory, f)) 
                       and f.startswith('CSMAXSCRT')]
    for CSMAXSCRTname in CSMAXSCRT_files:
        maxCohesiveCSMUCRT = MaxCohesiveCSMAXSCRT_Output(CSMAXSCRTname, 
                                                         dataDirectory, 
                                                         maxForceTime, 
                                                         dataCompare)
    return maxCohesiveCSMUCRT