# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:27:47 2020

@author: Kiffer2

"""
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patheffects as pe
import os
import os.path
import sys
import pdb

def Least_Squares(x, y):
    """
    Calculate the slope and y-intercept using matrix math
    x & y are the coordinates of points
    
    parameters (X,Y) Data
    
    Returns:
        Curve fit data and parameters m*x + b, R squared value
    """
    Z = np.ones((len(x),2))
    Z[:,1] = x
    # Calculate the matrix inverse for the constants of the regression
    A = np.dot(np.linalg.inv(np.dot(Z.T,Z)),(np.dot(Z.T,y)))
    linFit = x*A[1] + A[0]
    
    # Stats
    SS_tot = np.sum((y - np.mean(y))**2)
    SS_res = np.sum((y - linFit)**2)
    Rsqd = 1 - SS_res/SS_tot
    
    return linFit, A, Rsqd


def residualFcn(fileName, dataDirectory, maxForceTime, dataCompare, objErr, 
                slopeFlag, maxForceFlag, ssForceFlag, timeBeforePeak):
    """
    Parameters
    ----------
    fileName : Output txt file with the odb data
    dataDirectory : Location of the output file

    Returns
    -------
    Maximum force from the txt file
    """
    
    # In[Simulated data]
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
    Header.append('Retina_Glue_Top')
    Header.append('Bond_Displacements')
    
    df.columns = Header
    
    tt = df.Time
    RF = df.RF_y_dot*1e3 # Convert from N to mN
    NF = df.Nodal_Force*1e3 # Convert from N to mN
    CnF_RV = df.CnormF_RV*1e3 # Convert from N to mN
    CnF_VR = df.CnormF_VR*1e3 # Convert from N to mN
    Cp_RV = df.Cpress_RV
    Cp_VR = df.Cpress_VR
    AVG_Cp_RV = df.AVG_Cpress_RV_AVG
    AVG_Cp_VR = df.AVG_Cpress_VR_AVG
    Cs1_RV = df.Cshear1_RV*1e3 # Convert from N to mN
    Cs1_VR = df.Cshear1_VR*1e3 # Convert from N to mN
    Cs2_RV = df.Cshear2_RV*1e3 # Convert from N to mN
    Cs2_VR = df.Cshear2_VR*1e3 # Convert from N to mN
    CsF_RV = df.CshearF_RV*1e3 # Convert from N to mN
    CsF_VR = df.CshearF_VR*1e3 # Convert from N to mN
    dn = df.Retina_Glue_Top*1e3 # Convert from m to mm
    BD = df.Bond_Displacements*1e3 # Convert from m to mm
    
    # maybe try to output the maximum force at a specific time
    specificTime = maxForceTime
    actualTime = min(df['Time'], key=lambda x:abs(x - specificTime))
    force_at_time = RF[df['Time'] == actualTime].values[0]
    
    # In[Experimental data]
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
    HFSS =     float(dfValsn['Attribute'][17]) # (mN)
    # slope from 20 seconds prior to max force value
    HSlope20 = float(dfValsn['Attribute'][20]) # (mN/m)
    
    dfn = pd.read_csv(os.path.join(dataCompare), sep="\t", header=30)
    dfn.columns = ['Time', 'Extension', 'Force']
    dfn_time = dfn.Time
    dfn_extension = dfn.Extension # mm
    dfn_force = dfn.Force*1e3 # N ---> mN
    
    # if fileName.find('sym') >= 0:
    #     # divide all data trace values by 2
    #     dfn_force = dfn_force/2
    #     HFMax = HFMax/2
    #     HFSS = HFSS/2
    
    # SS Array
    ssTimeArray = np.array([HSSi, HSSf])
    ssValArray = np.array([HFSS, HFSS])
    
# In[Experimental data isolate linear region up to peak]
    
    # slope calculation for 20 seconds prior to the max peel force 
    # (Experimental Data)
    maxIndex = dfn_time[dfn_time == HTMax].index.values[0]
    
    # Convert to data array length
    timeBeforePeak = timeBeforePeak*10
    
    # Array from maxIndex - timeBeforePeak*10 (timeBeforePeak sec) to location of max force
    x_n = dfn_extension[maxIndex - timeBeforePeak:maxIndex]
    y = dfn_force[maxIndex - timeBeforePeak:maxIndex]
    # Perform least squares
    curveFit_n, Params_n, R_Squared_n = Least_Squares(x_n, y)
    
    # Shift extension data so that the linear region is extrapolated 
    # through the origin
    shift_disp = abs(Params_n[0]/Params_n[1])
    if Params_n[0] > 0:
        dfn_extension_shift = dfn_extension + shift_disp
        
        if min(dfn_extension_shift) > 0:
            # Add zero to prevent mishaps with interpolation
            dfn_extension_shift = [0] + dfn_extension_shift
    else:
        dfn_extension_shift = dfn_extension - shift_disp
        
    # Now that the data has been shifted, recalculate the linear regression 
    # using the reduced data set
    # Array from maxIndex - timeBeforePeak*10 (timeBeforePeak sec) to location of max force
    x_n = dfn_extension_shift[maxIndex - timeBeforePeak:maxIndex]
    # Array from maxIndex - timeBeforePeak*10 (timeBeforePeak sec) to location of max force
    y = dfn_force[maxIndex - timeBeforePeak:maxIndex]
    # Perform least squares
    curveFit_n, Params_n, R_Squared_n = Least_Squares(x_n,y)
    
    # Slope of the curve up to the max force !!!(from the simulated data)!!!
    # find the closest simulated displacement to the experimental 
    # max displacement
    # adjustDisp = min(dn, key=lambda x:abs(x - dfn_extension[maxIndex]))
    # index = RF[dn == adjustDisp].index.values[0] # index determination
    # Index where the max reaction force is in the array
    simMaxIndex = RF.idxmax()
    simMaxForce = RF.max() # maximum simulated force value
    simMaxDisp = dn[RF == simMaxForce] # displacement at the max force value
    
    # If the max index is the second data point add one to it (Difficulty in 
    # selecting the pandas series value) to select the fist two values in the 
    # pandas array it needs to be RF[0:2] instead of RF[0:1] but the index
    # value of the max force is 1.  Try to fix this issue
    if simMaxIndex == 1:
        simMaxIndex += 1
        
    x = dn[0:simMaxIndex] # Array from 0 to location of max force/n
    y = RF[0:simMaxIndex] # Array from 0 to location of max force/n
    # Perform least squares
    curveFit, Params, R_Squared = Least_Squares(x,y)
    
    # Updated force at specific max disp with adjusted value (Simulated data)
    specificTime = maxForceTime
    actualDisp = min(dn, key=lambda x:abs(x - dfn_extension_shift[maxIndex]))
    force_at_Disp = RF[dn == actualDisp].values[0]
    
    # Max peel force displacement at max and steady state
    dfn_max_Disp = dfn_extension_shift[dfn_time == HTMax]
    dfn_ss_Disp = [dfn_extension_shift[dfn_time == HSSi].values[0], 
                   dfn_extension_shift[dfn_time == HSSf].values[0]] # flatten()
    
    """ Simulated Steady State calculation """
    if simMaxIndex == len(RF):
        simMaxGreaterIndex = len(RF) - 1
    else:
        # return the mean and median of the points after the peak force value
        # This will always round down
        simMaxGreaterIndex = int(simMaxIndex + (len(RF) - simMaxIndex)*(31/64))
    
    # Steady state values from the max force index half way to the end
    # Force values after the peak force
    RF_SteadyState = RF[simMaxGreaterIndex:]
    # Displacement values after the peak force
    dn_SteadyState = dn[simMaxGreaterIndex:]
    
    SSMean = np.mean(RF_SteadyState) # Mean
    SSMedian = np.median(RF_SteadyState) # Median
    
    # In[Plots]
    """ Plots """
    # Plot the experimental, simulated, and curve fit data
    
    # Split the file extension
    (figureName, ext) = os.path.splitext(fileName)
    
    # Plot the data trace to compare the simulated results with the force 
    # displacement curves
    plt.plot(dfn_extension_shift, dfn_force,'-', color='r', linewidth=1, 
             markersize=2, label = '{}, Age: {}'.format(HID, HAGE), 
             alpha = 0.5)
    
    if str(HFMax) == 'nan' and str(HSSi) == 'nan':
        print('No max or steady state')
        pass
    
    if str(HFMax) != 'nan':
        plt.plot(dfn_max_Disp, HFMax,'.', color='k', linewidth=1, 
                 markersize=20, 
                 label = 'Max Peel - {:.4f} (mN)'.format(HFMax), 
                 path_effects=[pe.Stroke(linewidth=4, foreground='k'), 
                               pe.Normal()])
        plt.plot(x_n, curveFit_n, '-', color='tab:blue', linewidth=2, 
                 label=r'Curve fit Max - {} (s) '.format(timeBeforePeak/10) + 
                 'y = {:.4f}x + '.format(Params_n[1]) + 
                 '{:.4f} (mN), '.format(Params_n[0]) + 
                 '$r^2$ = {:.4f}'.format(R_Squared_n), alpha = 1)

    if str(HSSi) != 'nan':
        plt.plot(dfn_ss_Disp, ssValArray,'-', color='c', linewidth=3, 
                 markersize=2, 
                 label = 'Steady State - {:.4f} (mN)'.format(HFSS), 
                 path_effects=[pe.Stroke(linewidth=5, foreground='k'), 
                               pe.Normal()])
    
    # Plot the simulated data
    plt.plot(dn, RF,'-', color='blue', linewidth=2, markersize=2, 
             label = r'Simulated Reaction force $\Sigma F_{Retina}$')
    plt.plot(x, curveFit,'-', color='tab:green', linewidth=2, markersize=2, 
             label= 'y = {:.4f}x + '.format(Params[1]) + 
             '{:.4f} (mN), '.format(Params[0]) + 
             '$r^2$ = {:.4f}'.format(R_Squared))
    plt.plot(simMaxDisp, simMaxForce, '.', color='tab:red', linewidth=1, 
             markersize = 20, 
             label = 'Simulated maximum Force {:.4f} (mN)'.format(simMaxForce))
    plt.plot(dn_SteadyState, np.ones(len(RF_SteadyState))*SSMean, '-', 
             color='tab:gray', label = 'Simulated steady state force ' + 
             '{:.4f} (mN)'.format(np.mean(RF_SteadyState)))
    
    # In[Error Calculation]
    # error between slope, force, and steady-state value
    
    maxSlopeMeasured = Params_n[1] # Experimental slope
    maxSlopeSimulated = Params[1] # Simulated slope
    maxForceMeasured = HFMax # Experimental max force
    maxForceSimulated = simMaxForce # Simulated max force
    SS_Measured = HFSS # Experimental SS force
    SSmeanSimulated = SSMean # Simulated SS force (mean)
    SSmedianSimulated = SSMedian # Simulated SS force (median)
    
    # Error calculation
    errorDict = {} # Dictionary
    if objErr == 'Difference':
        errorDict['slope']    = (maxSlopeMeasured - maxSlopeSimulated) if slopeFlag == True else []
        errorDict['maxForce'] = (maxForceMeasured - maxForceSimulated) if maxForceFlag == True else []
        errorDict['ssForce']  = (SS_Measured - SSmeanSimulated)        if ssForceFlag == True else []
    elif objErr == 'Ratio':
        errorDict['slope']    = (1 - maxSlopeMeasured / maxSlopeSimulated) if slopeFlag == True else []
        errorDict['maxForce'] = (1 - maxForceMeasured / maxForceSimulated) if maxForceFlag == True else []
        errorDict['ssForce']  = (1 - SS_Measured / SSmeanSimulated)        if ssForceFlag == True else []
    elif objErr == 'Relative uncertainty':
        errorDict['slope']    = ((maxSlopeMeasured - maxSlopeSimulated)/maxSlopeMeasured) if slopeFlag == True else []
        errorDict['maxForce'] = ((maxForceMeasured - maxForceSimulated)/maxForceMeasured) if maxForceFlag == True else []
        errorDict['ssForce']  = ((SS_Measured - SSmedianSimulated)/SS_Measured)           if ssForceFlag == True else []
    else:
        print('Error in MaxForceError')
        sys.exit()
    
    # Error array values
    errorList = list(errorDict.values()) # convert to list
    errorList = [x for x in errorList if x] # get rid of empty values
    
    # String for the error array
    errorString = ', '.join('{:.4}'.format(i) for i in errorList)
    
    plt.plot([dfn_max_Disp, simMaxDisp], [HFMax, simMaxForce], '--', 
             linewidth = 1, color = 'magenta', label = r'Difference ' + 
             'between simulated & experiment max force: ' + 
             '{:.4f}'.format(HFMax - np.max(RF)))
    
    # Plot the different conditions if they are to be compared
    if slopeFlag == True:
        plt.plot([], [], 'white', label = r'{} '.format(objErr) + 
                 'between slopes is:  ' + 
                 '{:.4f}'.format(errorDict['slope']))
    
    if maxForceFlag == True:
        plt.plot([], [], 'white', label = r'{} '.format(objErr) + 
                 'between max force is:  ' + 
                 '{:.4f}'.format(errorDict['maxForce']))
    
    if ssForceFlag == True:
        plt.plot([], [], 'white', label = r'{} '.format(objErr) + 
                 'between steady state is:  ' + 
                 '{:.4f}'.format(errorDict['ssForce']))
        
    plt.plot([], [], 'white', 
             label = r'Objective error array:  [' + errorString + ']')
    plt.plot([], [], 'white', label = r'Error $L^2$ Norm: ' + 
             '{:.4f}'.format(np.sqrt(np.dot(errorList, errorList))))
    
    ################## Plot Data #########################
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.ylabel('Force (mN)',fontsize=18)
    plt.xlabel('Distance (mm)',fontsize=18)
    plt.title('Simulation vs. Experimental Data Trace',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + 
                             '_SlopeCompare.pdf'), dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    # In[Calculate interpolated Experimental and Simulated data]
    
    # slope calculation for 20 seconds prior to the max peel force 
    # (Experimental Data)
    maxIndex = dfn_time[dfn_time == HTMax].index.values[0]
    # Array from maxIndex - timeBeforePeak*10 (timeBeforePeak sec) to location of max force
    t_n = dfn_time[maxIndex - timeBeforePeak:maxIndex]
    y = dfn_force[maxIndex - timeBeforePeak:maxIndex]
    # Perform least squares and return
    curveFit_n, Params_n_time, R_Squared_n = Least_Squares(t_n, y)
    
    # Shift extension data so that the linear region is extrapolated 
    # through the origin
    shift_time = abs(Params_n_time[0]/Params_n_time[1])
    
    # shift time data for visual purposes
    if Params_n_time[0] > 0:
        dfn_time_shift = dfn_time + shift_time
        
        if min(dfn_time_shift) > 0:
            # Add zero to prevent mishaps with interpolation
            dfn_time_shift = [0] + dfn_time_shift
    else:
        dfn_time_shift = dfn_time - shift_time
    
    # x array for the linear region leading up to the peak force
    Fmax_t_shift = dfn_time_shift[maxIndex]
    fit_t = np.linspace(0, Fmax_t_shift, 200) # Selected value
    # fit_t = np.linspace(0, dfn_time_shift[np.argmax(dfn_force)], 200) # true max
    Fmax_x_shift = dfn_extension_shift[maxIndex]
    # fit_x = np.linspace(0, dfn_extension_shift[np.argmax(dfn_force)], 200) # true max
    fit_x = np.linspace(0, Fmax_x_shift, 200) # Selected value
    
    # create the linear region leading up to the peak force
    def fit(params, x):
        b, m = params
        return m*x + b
    fit_vals_y_time = fit(Params_n_time, fit_t)
    fit_vals_y_force = fit(Params_n, fit_x)
    
    # Trim the shifted experimental data to be greater than zero
    t_exp = dfn_time_shift[dfn_time_shift >= 0]
    x_exp = dfn_extension_shift[dfn_time_shift >= 0]
    y_exp = dfn_force[dfn_time_shift >= 0]
    
    # data frame with original data only shifted
    dfdata = pd.DataFrame(np.array([t_exp, x_exp, y_exp]).T, 
                          columns=['t', 'x', 'y'])
    
    # Select time beyond the max time to the end of the data
    t_geq_max = dfn_time_shift[maxIndex:]
    x_geq_max = dfn_extension_shift[maxIndex:]
    y_geq_max = dfn_force[maxIndex:]
    
    # dataframe of data points from the max value to the end
    dfgmax = pd.DataFrame(np.array([t_geq_max, x_geq_max, y_geq_max]).T, 
                          columns=['t', 'x', 'y'])
    
    # data frame of points from zero to the max value
    linArray = np.array([fit_t, fit_x, fit_vals_y_force])
    dfLin = pd.DataFrame(linArray.T, columns=['t', 'x', 'y'])
    
    # create the new data frame of linear points up to the peak and all points 
    # beyond
    dfNew = dfLin.append(dfgmax, ignore_index=True)
    
    # Interpolate the experimental data
    n_data_pts = 100
    start_point_time = tt[RF.argmax()] # Time at the peak (simulated)
    start_point_disp = dn[RF.argmax()] # Disp at the peak (simulated)
    f_exp_time = interpolate.interp1d(dfNew['t'], dfNew['y'])
    f_exp_disp = interpolate.interp1d(dfNew['x'], dfNew['y'])
    t_new_exp = np.linspace(start_point_time, tt[tt.argmax()], 
                            n_data_pts) # (s)
    x_new_exp = np.linspace(start_point_disp, dn[tt.argmax()], 
                            n_data_pts) # (mm)
    y_new_exp_time = f_exp_time(t_new_exp) # Interpolate `interp1d`
    y_new_exp_disp = f_exp_disp(x_new_exp) # Interpolate `interp1d`
    
    # In[Interpolated Simulated Trace]
    
    # Interpolate the simulated data
    f_sim_time = interpolate.interp1d(tt, RF)
    f_sim_disp = interpolate.interp1d(dn, RF)
    t_new_sim = np.linspace(start_point_time, tt[tt.argmax()], 
                            n_data_pts) # (s)
    x_new_sim = np.linspace(start_point_disp, dn[tt.argmax()], 
                            n_data_pts) # (mm)
    y_new_sim_time = f_sim_time(t_new_sim) # Interpolate `interp1d`
    y_new_sim_disp = f_sim_disp(x_new_sim) # Interpolate `interp1d`
    
    # In[Plots]
    ''' Time curve '''
    fit, ax = plt.subplots()
    ax.plot()
    ax.plot(dfdata['t'], dfdata['y'], label='Original Shifted Data', 
            alpha = 0.5)
    ax.plot(dfNew['t'], dfNew['y'], label='Merged Data', 
            alpha = 0.5)
    ax.plot(t_new_exp, y_new_exp_time, '--', label='Interp Experimental Data')
    ax.plot(tt, RF, label='Simulated Data')
    ax.plot(t_new_sim, y_new_sim_time, ':', label='Interp Simulated Data')
    ax.axhline(color='k')
    ax.set_xlim([0, 300])
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Force (N)', fontsize=14)
    ax.legend(loc='best', fontsize=14)
    ax.grid('on')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + 
                             '_Interp_Time.pdf'), dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    ''' Displacement curve '''
    fit, ax = plt.subplots()
    ax.plot()
    ax.plot(dfdata['x'], dfdata['y'], label='Original Shifted Data', 
            alpha = 0.5)
    ax.plot(dfNew['x'], dfNew['y'], label='Merged Data', 
            alpha = 0.5)
    ax.plot(x_new_exp, y_new_exp_disp, '--', label='Interp Experimental Data')
    ax.plot(dn, RF, label='Simulated Data')
    ax.plot(x_new_sim, y_new_sim_disp, ':', label='Interp Simulated Data')
    ax.axhline(color='k')
    ax.set_xlim([0, max(dn)])
    ax.set_xlabel('Displacement (mm)', fontsize=14)
    ax.set_ylabel('Force (N)', fontsize=14)
    ax.legend(loc='best', fontsize=14)
    ax.grid('on')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + 
                             '_Interp_Disp.pdf'), dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    ''' Displacement curve only showing interpolated data '''
    residual = y_new_exp_disp - y_new_sim_disp # residual calculation
    L2Norm = np.sqrt(np.dot(residual, residual))
    
    fit, ax = plt.subplots()
    ax.plot()
    ax.plot(x_new_exp, y_new_exp_disp, '-', label='Interp Experimental Data')
    ax.plot(x_new_sim, y_new_sim_disp, '-', label='Interp Simulated Data')
    ax.plot(x_new_sim, residual, ':', label=r'Residual = $(exp - sim)$',
            alpha = 0.8)
    ax.plot([], [], color='white', label=r'$L^2$ norm = {:.4f}'.format(L2Norm))
    ax.axhline(color='k', linewidth=0.25)
    ax.set_xlim([0, max(x_new_exp)])
    ax.set_xlabel('Displacement (mm)', fontsize=14)
    ax.set_ylabel('Force (N)', fontsize=14)
    ax.legend(loc='best', fontsize=14)
    ax.grid('on')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + 
                             '_Interp_Disp_Clean.pdf'), dpi=300, 
                bbox_inches='tight')
    plt.close()
    
    returnList = [Params[1], simMaxForce, SSMean, SSMedian, y_new_exp_disp, 
                  y_new_sim_disp]
    return returnList

# In[Function that calls the nested function to compute the residual]
def findResidual(fileName, dataDirectory, maxForceTime, dataCompare, objErr, 
                 slopeFlag, maxForceFlag, ssForceFlag, timeBeforePeak):
    """
    Parameters
    ----------
    fileName : Output txt file with the odb data
    dataDirectory : Location of the output file

    Returns
    -------
    maximumForce : Maximum force from the txt file
    """
    
    global residual
    """ Call function to return max displacement """
    ModelParamsFile = [f for f in os.listdir(dataDirectory) 
                       if os.path.isfile(os.path.join(dataDirectory, f)) 
                       and f.startswith('output_Field')]
    for mpFile in ModelParamsFile:
        residual = residualFcn(mpFile, dataDirectory, maxForceTime, 
                               dataCompare, objErr, slopeFlag, maxForceFlag, 
                               ssForceFlag, timeBeforePeak)
    
    return residual