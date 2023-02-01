# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 23:56:35 2021

@author: Kiffer2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patheffects as pe
import os
import os.path
import sys
import pdb

def Least_Squares(x,y):
    """
    Calculate the slope and y-intercept using matrix math
    x & y are the coordinates of points
    
    parameters (X,Y) Data
    
    Returns:
        Curve fit data and parameters m*x + b, R squared value
    """
    Z = np.ones((len(x),2))
    Z[:,1] = x
    A = np.dot(np.linalg.inv(np.dot(Z.T,Z)),(np.dot(Z.T,y))) # Calculate the matrix inverse for the constants of the regression
    linFit = x*A[1] + A[0]
    
    # Stats
    SS_tot = np.sum((y - np.mean(y))**2)
    SS_res = np.sum((y - linFit)**2)
    Rsqd = 1 - SS_res/SS_tot
    
    return linFit, A, Rsqd

def maxForce(fileName, dataDirectory, maxForceTime, dataCompare):
    """
    Parameters
    ----------
    fileName : Output txt file with the odb data
    dataDirectory : Location of the output file

    Returns
    -------
    Maximum force from the txt file
    """
    df = pd.read_csv(os.path.join(dataDirectory, fileName), sep="\t", header=0)
    
    Header = [] # Header information for the dataframe
    Header.append('Frame')
    Header.append('Time')
    Header.append('RF_y_dot')
    Header.append('RFx')
    Header.append('RFy')
    Header.append('RFz')
    Header.append('Nodal_Force')
    Header.append('CnormF_RV')
    Header.append('CnormF_VR')
    Header.append('Cpress_RV')
    Header.append('Cpress_VR')
    Header.append('AVG_Cpress_RV_AVG')
    Header.append('AVG_Cpress_VR_AVG')
    Header.append('Cshear1_RV')
    Header.append('Cshear1_VR')
    Header.append('Cshear2_RV')
    Header.append('Cshear2_VR')
    Header.append('CshearF_RV')
    Header.append('CshearF_VR')
    Header.append('Glue_Displacements')
    Header.append('Bond_Displacements')
    
    df.columns = Header
    
    RF = df.RF_y_dot*1e3 # N to mN
    dn = df.Glue_Displacements*1e3 # m to mm
    
    # maybe try to output the maximum force at a specific time
    specificTime = maxForceTime
    actualTime = min(df['Time'], key=lambda x:abs(x - specificTime))
    force_at_time = RF[df['Time'] == actualTime].values[0]
    
    # Plot the experimental, simulated, and curve fit data
    
    (figureName, ext) = os.path.splitext(fileName) # Split the file extension
    
    """ Read in the csv file """
    dfValsn = pd.read_csv(os.path.join(dataCompare), sep="\t", nrows=29, header=None, names=['Var', 'Attribute'])
    
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
    HSlope20 = float(dfValsn['Attribute'][20]) # (mN/m) slope from 20 seconds prior to max force value
    
    dfn = pd.read_csv(os.path.join(dataCompare), sep="\t", header=30)
    dfn.columns = ['Time', 'Extension', 'Force']
    dfn_time = dfn.Time
    dfn_extension = dfn.Extension # mm
    dfn_force = dfn.Force*1e3 # N ---> mN
    
    # SS Array
    ssTimeArray = np.array([HSSi, HSSf])
    ssValArray = np.array([HFSS, HFSS])
    
    # slope calculation for 20 seconds prior to the max peel force (Experimental Data)
    maxIndex = dfn_time[dfn_time == HTMax].index.values[0]
    x20 = dfn_extension[maxIndex-200:maxIndex] # Array from maxIndex - 200 (20 sec) to location of max force
    y = dfn_force[maxIndex-200:maxIndex] # Array from maxIndex - 200 (20 sec) to location of max force
    curveFit20, Params20, R_Squared20 = Least_Squares(x20,y) # Perform least squares and return
    
    # Shift extension data so that the linear region is extrapolated through the origin
    shift = abs(Params20[0]/Params20[1])
    dfn_extension = dfn_extension - shift
    
    # Now that the data has been shifted, recalculate the linear regression using the reduced data set
    x20 = dfn_extension[maxIndex-200:maxIndex] # Array from maxIndex - 200 (20 sec) to location of max force
    y = dfn_force[maxIndex-200:maxIndex] # Array from maxIndex - 200 (20 sec) to location of max force
    curveFit20, Params20, R_Squared20 = Least_Squares(x20,y) # Perform least squares and return
    
    # Slope of the curve up to the max force !!!(from the simulated data)!!!
    adjustDisp = min(dn, key=lambda x:abs(x - dfn_extension[maxIndex]))
    index = RF[dn == adjustDisp].index.values[0]
    simulationCriteria = index # Time before peak force for curve fitting
    x = dn[index - simulationCriteria:index] # Array from 0 to location of max force
    y = RF[index - simulationCriteria:index] # Array from 0 to location of max force
    curveFit, Params, R_Squared = Least_Squares(x,y) # Perform least squares and return
    
    # Updated force at specific max disp with adjusted value (Simulated data)
    specificTime = maxForceTime
    actualDisp = min(dn, key=lambda x:abs(x - dfn_extension[maxIndex]))
    force_at_Disp = RF[dn == actualDisp].values[0]
    
    # Simulated max force
    simMaxForce = RF.max() # maximum simulated force value
    simMaxDisp = dn[RF == simMaxForce] # displacement at the max force value
    
    # Max peel force displacement at max and steady state
    dfn_max_Disp = dfn_extension[dfn_time == HTMax]
    # dfn_ss_Disp = np.array([dfn_extension[dfn_time == HSSi], dfn_extension[dfn_time == HSSf]]).flatten() # Didn't seem to work here
    dfn_ss_Disp = [dfn_extension[dfn_time == HSSi].values[0], dfn_extension[dfn_time == HSSf].values[0]]
    
    """ Plots """
    # Plot the data trace to compare the simulated results with the force displacement curves
    plt.plot(dfn_extension, dfn_force,'-', color='r', linewidth=1, markersize=2, label = '{}, Age: {}'.format(HID, HAGE), alpha = 0.5)
    
    if str(HFMax) == 'nan' and str(HSSi) == 'nan':
        print('No max or steady state')
        pass
    
    if str(HFMax) != 'nan':
        plt.plot(dfn_max_Disp, HFMax,'.', color='k', linewidth=1, markersize=20, label = 'Max Peel - {:.4f} (mN)'.format(HFMax), path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
        plt.plot(x20, curveFit20, '-', color='tab:blue', linewidth=2, label=r'Curve fit Max - 20 (s) y = {:.4f}x + {:.4f} (mN), $r^2$ = {:.4f}'.format(Params20[1], Params20[0], R_Squared20), alpha = 1)

    if str(HSSi) != 'nan':
        plt.plot(dfn_ss_Disp, ssValArray,'-', color='c', linewidth=3, markersize=2, label = 'Steady State - {:.4f} (mN)'.format(HFSS), path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
    
    plt.plot(dn, RF,'-',color='blue',linewidth=2,markersize=2,label = r'Simulated Reaction force $\Sigma F_{Retina}$')
    plt.plot(x, curveFit,'-', color='tab:green', linewidth=2, markersize=2, label= 'y = {:.4f}x + {:.4f} (mN), $r^2$ = {:.4f}'.format(Params[1], Params[0], R_Squared))
    plt.plot(actualDisp, force_at_Disp, '.', color='tab:orange', linewidth=1, markersize = 20, label = 'Force at max disp {:.4f} (mN)'.format(force_at_Disp))
    plt.plot(simMaxDisp, simMaxForce, '.', color='tab:red', linewidth=1, markersize = 20, label = 'Simulated maximum Force {:.4f} (mN)'.format(simMaxForce))
    
    # error between slope and force value
    plt.plot([actualDisp, dfn_max_Disp], [force_at_Disp, HFMax], '--', linewidth = 1, color = 'magenta', label = r'ABS difference between force @ peak values is: {:.4f}'.format(abs(HFMax - force_at_Disp)))
    plt.plot([], [], 'white', label = r'ABS difference between slopes is: {:.4f}'.format(abs(Params20[1] - Params[1])))
    plt.plot([], [], 'white', label = r'ABS ratio between slopes is: {:.4f}'.format(abs(Params20[1] / Params[1])))
    
    ################## Plot Data #########################
    plt.axhline(0,color='black') # x = 0
    plt.axvline(0,color='black') # y = 0    plt.xlabel('Displacement (mm)',fontsize=18)
    plt.ylabel('Force (mN)',fontsize=18)
    plt.title('Vitreous',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + '_SlopeCompare.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    return Params[1], force_at_Disp, np.max(RF) # Slope, force @ specified time, max force

def findMaxForce(fileName, dataDirectory, maxForceTime, dataCompare):
    """
    Parameters
    ----------
    fileName : Output txt file with the odb data
    dataDirectory : Location of the output file

    Returns
    -------
    maximumForce : Maximum force from the txt file
    """
    
    global maximumForce
    """ Call function to return max displacement """
    ModelParamsFile = [f for f in os.listdir(dataDirectory) if os.path.isfile(os.path.join(dataDirectory, f)) and f.startswith('output_Field')]
    for mpFile in ModelParamsFile:
        maximumForce = maxForce(mpFile, dataDirectory, maxForceTime, dataCompare)
    
    return maximumForce