# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:07:50 2021

@author: Kiffer2
"""

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon
plt.rcParams['figure.figsize'] = [16, 9]
from scipy import interpolate
import pdb


# # Define the location of the Abaqus Working Directory
# # specific folder path where this file is located
# pythonScriptPath = os.getcwd()
# abqWD, pythonFiles = os.path.split(pythonScriptPath) # split file path

# filePath = os.getcwd() # current working directory
# codePath, pythonFolder = os.path.split(filePath) # split file path
# HWPath, codesFolder = os.path.split(codePath) # split file path

# expDataPath = 'experimentalData' # folder of data files
# dataPath = os.path.join(HWPath, expDataPath) # Path to data files

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
    # Calculate the matrix inverse for the constants of the regression
    A = np.dot(np.linalg.inv(np.dot(Z.T,Z)),(np.dot(Z.T,y)))
    linFit = x*A[1] + A[0]
    
    # Stats
    SS_tot = np.sum((y - np.mean(y))**2)
    SS_res = np.sum((y - linFit)**2)
    Rsqd = 1 - SS_res/SS_tot
    
    return linFit, A, Rsqd

def myformat(x):
    myexp = int(np.floor(np.log10(x)))
    xout = x*10**(-myexp)
    strout = '{:.4f}'.format(xout) + '\cdot10^{' + '{}'.format(myexp) + '}'
    return strout


# In[previous data]

def ReadRAWDataTrace(dataPath, abqWD, timeBeforePeak):
    """
    Inputs: dataPath - file path to raw data
    abqWD: abaqus working directory
    timeBeforePeak: number of seconds prior to the peak where data will 
                    be extrapolated to the origin for curve fitting
    """
    
    timeBeforePeak = timeBeforePeak*10 # Convert s --> cs (10 data points/sec)
    
    # Eliminate the file extension
    dataPathNoExt = dataPath.split('.txt')[0]
    
    # Determine the specific file name
    fileDir, dataCompare = os.path.split(dataPathNoExt)
    
    """ Read in the csv file """
    dfValsn = pd.read_csv(dataPath, sep="\t", nrows=29, header=None, 
                          names=['Var', 'Attribute'])
    
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
    # slope from 20 seconds prior to max force value
    HSlope20 = float(dfValsn['Attribute'][20]) # (mN/m)
    
    dfn = pd.read_csv(dataPath, sep="\t", header=30)
    dfn.columns = ['Time', 'Extension', 'Force']
    dfn_time = dfn.Time
    dfn_extension = dfn.Extension # mm
    dfn_force = dfn.Force*1e3 # N ---> mN
    
    # SS Array
    ssTimeArray = np.array([HSSi, HSSf])
    ssValArray = np.array([HFSS, HFSS])
    
    # slope calculation for 20 seconds prior to the max peel force 
    # (Experimental Data)
    maxIndex = dfn_time[dfn_time == HTMax].index.values[0]
    
    # to location of max force
    # Array from maxIndex - timeBeforePeak*10 (timeBeforePeak sec) 
    t_n = dfn_extension[maxIndex - timeBeforePeak:maxIndex]
    # to location of max force
    # Array from maxIndex - timeBeforePeak*10 (timeBeforePeak sec) 
    y = dfn_force[maxIndex - timeBeforePeak:maxIndex]
    # Perform least squares and return
    curveFit_n, Params_n, R_Squared_n = Least_Squares(t_n,y)
    
    # Shift extension data so that the linear region is extrapolated through 
    # the origin
    shift = abs(Params_n[0]/Params_n[1])*0
    dfn_extension = dfn_extension - shift
    
    # Now that the data has been shifted, recalculate the linear regression 
    # using the reduced data set
    
    # to location of max force
    # Array from maxIndex - timeBeforePeak*10 (timeBeforePeak sec)
    t_n = dfn_extension[maxIndex - timeBeforePeak:maxIndex]
    # to location of max force
    # Array from maxIndex - timeBeforePeak*10 (timeBeforePeak sec)
    y = dfn_force[maxIndex - timeBeforePeak:maxIndex]
    # Perform least squares and return
    curveFit_n, Params_n, R_Squared_n = Least_Squares(t_n,y)
    
    # # Slope of the curve up to the max force !!!(from the simulated data)!!!
    # adjustDisp = min(dn, key=lambda x:abs(x - dfn_extension[maxIndex]))
    # index = RF[dn == adjustDisp].index.values[0]
    # simulationCriteria = index # Time before peak force for curve fitting
    # # Array from 0 to location of max force
    # x = dn[index - simulationCriteria:index]
    # # Array from 0 to location of max force
    # y = RF[index - simulationCriteria:index]
    # # Perform least squares
    # curveFit, Params, R_Squared = Least_Squares(x,y) 
    
    # # Updated force at specific max disp with adjusted value (Simulated data)
    # specificTime = maxForceTime
    # actualDisp = min(dn, key=lambda x:abs(x - dfn_extension[maxIndex]))
    # force_at_Disp = RF[dn == actualDisp].values[0]
    
    # # Simulated max force
    # simMaxForce = RF.max() # maximum simulated force value
    # simMaxDisp = dn[RF == simMaxForce] # displacement at the max force value
    
    # Max peel force displacement at max and steady state
    dfn_max_Disp = dfn_extension[dfn_time == HTMax]
    # Didn't seem to work here
    # dfn_ss_Disp = np.array([dfn_extension[dfn_time == HSSi], 
    #                         dfn_extension[dfn_time == HSSf]]).flatten()
    dfn_ss_Disp = [dfn_extension[dfn_time == HSSi].values[0], 
                   dfn_extension[dfn_time == HSSf].values[0]]
    
    # In[Simulated Trace]
    
    # dataDirectory = 'D:\Downloads\experimentalData'
    
    # fileName = ('output_Field_S25CohesiveXLVitDiff_CT250S11' + 
    #             'SF0MS7RE1e_04VE5e_02opt.txt')
    
    # df = pd.read_csv(os.path.join(dataDirectory, fileName), 
    #                  sep="\t", header=0)
        
    # Header = [] # Header information for the dataframe
    # Header.append('Frame') #                   h1
    # Header.append('Time') #                    h2
    # Header.append('RF_y_dot') #                h3
    # Header.append('RFx') #                     h4
    # Header.append('RFy') #                     h5
    # Header.append('RFz') #                     h6
    # Header.append('Nodal_Force') #             h7
    # Header.append('Tab_Displacement') #        h8
    # Header.append('Bond_Displacement') #       h9
    # Header.append('Stress') #                  h10
    # Header.append('AVG_CSMAXSCRT') #           h11
    # Header.append('AVG_CSDMG') #               h12
    # df.columns = Header
    
    # tt = df.Time
    # RF = df.RF_y_dot*1000 # N to mN
    # dn = df.Tab_Displacement*1000 # m
    
    # In[Plots]
    
    """ Plots """
    # Plot the data trace to compare the simulated results with the force 
    # displacement curves
    fig, ax = plt.subplots()
    ax.plot(dfn_extension, dfn_force,'-', color='r', linewidth=1, 
            markersize=2, label = '{}, Age: {}'.format(HID, HAGE), 
            alpha = 0.5)
    
    if str(HFMax) == 'nan' and str(HSSi) == 'nan':
        print('No max or steady state')
        pass
    
    if str(HFMax) != 'nan':
        ax.plot(dfn_max_Disp, HFMax,'.', color='k', linewidth=1, 
                markersize=20, 
                label = 'Max Peel - {:.4f} (mN)'.format(HFMax), 
                path_effects=[pe.Stroke(linewidth=4, foreground='k'), 
                              pe.Normal()])
        ax.plot(t_n, curveFit_n, '-', color='tab:blue', linewidth=2, 
                label=r'Curve fit Max - {}'.format(int(timeBeforePeak/10)) + 
                ' (s) y = {:.4f}x '.format(Params_n[1]) + 
                '+ {:.4f} (mN), '.format(Params_n[0]) + 
                '$r^2$ = {:.4f}'.format(R_Squared_n), 
                alpha = 1)
    
    if str(HSSi) != 'nan':
        ax.plot(dfn_ss_Disp, ssValArray,'-', color='c', linewidth=3, 
                markersize=2, 
                label = 'Steady State - {:.4f} (mN)'.format(HFSS), 
                path_effects=[pe.Stroke(linewidth=5, 
                                        foreground='k'), 
                              pe.Normal()])
    
    # Make the shaded region for the entire integral
    a = dfn_max_Disp.values[0] # dfn_ss_Disp[0]
    b = dfn_ss_Disp[0] # dfn_ss_Disp[1]
    
    # Make the shaded region include the square below
    adjust = 0 # 0 or 1 to get rid of the small square
    
    # Filter the data in between the bounds
    dfn_ext_adjust = dfn_extension[(dfn_extension >= a) & (dfn_extension < b)]
    dnf_force_adjust = dfn_force[(dfn_extension >= a) & (dfn_extension < b)]
    
    verts = [(a, HFSS*adjust), 
             *zip(dfn_ext_adjust, dnf_force_adjust), 
             (b, HFSS*adjust)]
    poly = Polygon(verts, facecolor='0.8', edgecolor='0.5')
    ax.add_patch(poly)
    
    # Integral area
    Integral = np.trapz(dnf_force_adjust - HFSS*adjust, dfn_ext_adjust)
    
    # Centroid for plotting
    CentroidX = 1/Integral*(np.trapz(dfn_ext_adjust*(dnf_force_adjust - 
                                                     HFSS*adjust), 
                                     dfn_ext_adjust))
    CentroidY = 1/Integral*(np.trapz((dnf_force_adjust**2 - 
                                      (HFSS*adjust)**2*adjust)/2, 
                                     dfn_ext_adjust))
    
    # ax.text(b, (HFMax + HFSS)/2, r'$\int_a^b f(x)\mathrm{d}x=' + 
    #         myformat(Integral*1e-6) + '$ (J)', horizontalalignment='center', 
    #         fontsize=20)
    # ax.plot([0.5*max(dfn_extension), CentroidX], [0.5*max(dfn_force), 
    #                                               CentroidY])
    
    prop = dict(arrowstyle="-|>,head_width=0.4, head_length=0.8", shrinkA=0, 
                shrinkB=0)
    # ax.arrow(0.5*max(dfn_extension), 0.5*max(dfn_force), 
    #          CentroidX - 0.5*max(dfn_extension), 
    #          CentroidY - 0.5*max(dfn_force), 
    #          head_width=0.1, head_length=0.1)
    ax.annotate("", xy=(CentroidX, CentroidY), xytext=(0.5*max(dfn_extension), 
                                                       0.5*max(dfn_force)), 
                arrowprops=prop)
    
    ax.text(0.5*max(dfn_extension), 0.52*max(dfn_force), 
            r'$\int_a^b f(x)\mathrm{d}x=' + myformat(Integral*1e-6) + '$ (J)', 
            horizontalalignment='center', fontsize=20)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    
    ax.set_xticks((a, b))
    ax.set_xticklabels(('${}$'.format(a), '${}$'.format(b)))
    ax.set_yticks((HFSS, HFMax))
    ax.set_yticklabels(('${:.5}$'.format(HFSS), '${:.5}$'.format(HFMax)))
    
    ################## Plot Data #########################
    plt.axhline(0,color='black') # x = 0
    plt.axvline(0,color='black') # y = 0
    plt.ylabel('Force (mN)',fontsize=18)
    plt.xlabel('Displacement (mm)',fontsize=18)
    plt.title('Vitreous',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(abqWD, 'GcSelection.pdf'), dpi=300, 
                bbox_inches='tight')
    # plt.show()
    plt.close()
    
    
    
    # """ Derivative of the data trace """
    # fig, ax = plt.subplots()
    
    # deriv = np.gradient(dfn_force, dfn_extension)
    
    # ax.plot(dfn_extension, deriv)
    # ax.set_ylim(-100, 100) # maxRFList
    # plt.show()
    
    # In[Time plot]
    
    # slope calculation for n seconds prior to the max peel force 
    # (Experimental Data)
    maxIndex = dfn_time[dfn_time == HTMax].index.values[0]
    
    # to location of max force
    # Array from maxIndex - timeBeforePeak*10 (timeBeforePeak sec) 
    t_n = dfn_time[maxIndex - timeBeforePeak:maxIndex]
    y = dfn_force[maxIndex - timeBeforePeak:maxIndex]
    # Perform least squares and return
    curveFit_n, Params_n, R_Squared_n = Least_Squares(t_n, y)
    
    # Shift extension data so that the linear region is extrapolated 
    # through the origin
    shift_time = abs(Params_n[0]/Params_n[1])*1
    if Params_n[0] > 0:
        # shift time data for visual purposes
        dfn_time_shift = dfn_time + shift_time
        dfn_ss_time_shift = ssTimeArray + shift_time
        HTMax_shift = HTMax + shift_time
    else:
        # shift time data for visual purposes
        dfn_time_shift = dfn_time - shift_time
        dfn_ss_time_shift = ssTimeArray - shift_time
        HTMax_shift = HTMax - shift_time

    
    # Curve fit the shifted displacement
    maxIndex = dfn_time[dfn_time == HTMax].index.values[0]
    
    # to location of max force
    # Array from maxIndex - timeBeforePeak*10 (timeBeforePeak sec) 
    t_n = dfn_time_shift[maxIndex - timeBeforePeak:maxIndex]
    y = dfn_force[maxIndex - timeBeforePeak:maxIndex]
    # Perform least squares and return
    curveFit_n, Params_n, R_Squared_n = Least_Squares(t_n, y)
    
    # to location of max force
    # Array from maxIndex - timeBeforePeak*10 (timeBeforePeak sec) 
    x_n = dfn_extension[maxIndex - timeBeforePeak:maxIndex]
    # Perform least squares
    curveFit_n_disp, Params_n_disp, R_Squared_n_disp = Least_Squares(x_n, y)
    
    # Shift extension data so that the linear region is extrapolated through 
    # the origin
    shift_disp = abs(Params_n_disp[0]/Params_n_disp[1])*1
    if Params_n[0] > 0:
        dfn_extension_shift = dfn_extension + shift_disp
        dfn_ss_Disp_shift = dfn_ss_Disp + shift_disp
    else:
        dfn_extension_shift = dfn_extension - shift_disp
        dfn_ss_Disp_shift = dfn_ss_Disp - shift_disp
    
    # to location of max force
    # Array from maxIndex - timeBeforePeak*10 (timeBeforePeak sec)
    x_n = dfn_extension_shift[maxIndex - timeBeforePeak:maxIndex]
    # Perform least squares
    curveFit_n_disp, Params_n_disp, R_Squared_n_disp = Least_Squares(x_n, y)
    
    
    Fmax_t_shift = dfn_time_shift[maxIndex]
    fit_t = np.linspace(0, Fmax_t_shift, 200) # Selected value
    
    # true max
    # fit_t = np.linspace(0, dfn_time_shift[np.argmax(dfn_force)], 200)
    Fmax_x_shift = dfn_extension_shift[maxIndex]
    
    # true max
    # fit_x = np.linspace(0, dfn_extension_shift[np.argmax(dfn_force)], 200)
    fit_x = np.linspace(0, Fmax_x_shift, 200) # Selected value
    
    def fit(params, x):
        b, m = params
        return m*x + b
    
    fit_vals_y_time = fit(Params_n, fit_t)
    fit_vals_y_force = fit(Params_n_disp, fit_x)
    
    ''' Reaction force vs. time shifted '''
    fig, ax = plt.subplots()
    ax.plot(dfn_time_shift, dfn_force, 
            label=r'Data - {}'.format(dataCompare.split('.')[0]))
    ax.plot(fit_t, fit_vals_y_time, '--', label=r'Assumed linear region')
    ax.plot(Fmax_t_shift, dfn_force[maxIndex], 'o', markersize=10, 
            label=r'Time at peak = {:.4} (s)'.format(max(fit_t)))
    
    ax.plot(dfn_ss_time_shift, ssValArray,'-', color='c', linewidth=3, 
            markersize=2, label = 'Steady State - {:.4f} (mN)'.format(HFSS), 
            path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
    
    ax.plot([], [], 'w', 
            label='Start SS time = {:.4f} (s)'.format(min(dfn_ss_time_shift)))
    ax.plot([], [], 'w', 
            label='End SS time = {:.4f} (s)'.format(max(dfn_ss_time_shift)))
    
    plt.axhline(0,color='black')
    plt.axvline(0,color='black')
    
    plt.ylabel('Force (mN)',fontsize=18)
    plt.xlabel('Time from extrapolated zero (s)',fontsize=18)
    plt.legend(loc='best')
    # plt.xlim([0, max(dfn_time_shift)])
    plt.savefig(os.path.join(abqWD, 'SimulationTime.pdf'), dpi=300, 
                bbox_inches='tight')
    # plt.show()
    plt.close()
    
    ''' Reaction force vs. displacement shifted '''
    fig, ax = plt.subplots()
    ax.plot(dfn_extension_shift, dfn_force, 
            label=r'Data - {}'.format(dataCompare.split('.')[0]))
    ax.plot(fit_x, fit_vals_y_force, '--', label=r'Assumed linear region')
    ax.plot(Fmax_x_shift, dfn_force[maxIndex], 'o', markersize=10, 
            label=r'Time at peak = {:.4} (s)'.format(max(fit_t)))
    
    ax.plot(dfn_ss_Disp_shift, ssValArray,'-', color='c', linewidth=3, 
            markersize=2, label = 'Steady State - {:.4f} (mN)'.format(HFSS), 
            path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
    
    ax.plot([], [], 'w', 
            label='Start SS time = {:.4f} (s)'.format(min(dfn_ss_time_shift)))
    ax.plot([], [], 'w', 
            label='End SS time = {:.4f} (s)'.format(max(dfn_ss_time_shift)))
    
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    
    plt.ylabel('Force (mN)',fontsize=18)
    plt.xlabel('Displacement (mm)',fontsize=18)
    plt.legend(loc='best')
    # plt.xlim([0, max(dfn_time_shift)])
    plt.savefig(os.path.join(abqWD, 'SimulationDisp.pdf'), dpi=300, 
                bbox_inches='tight')
    # plt.show()
    plt.close()
    
    # In[Interpolated Experimental Data]
    
    # create array from 0 max peel force (linear equation fit from above)
    # populate a pandas dataframe
    # merge the data frame with the data above from the peak force to the end
    # use the interp1d fcn to interpolate between data
    # pass the simulated data into the interpolation
    
    # Time greater than the shift intersection point
    t_exp = dfn_time_shift[dfn_time_shift >= 0]
    x_exp = dfn_extension_shift[dfn_time_shift >= 0]
    y_exp = dfn_force[dfn_time_shift >= 0]
    
    # data frame with original data
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
    linArray = np.array([fit_t, fit_x, fit_vals_y_time])
    dfLin = pd.DataFrame(linArray.T, columns=['t', 'x', 'y'])
    
    # create the new data frame of linear points up to the peak and all points 
    # beyond
    dfNew = dfLin.append(dfgmax, ignore_index=True)
    
    # # Interpolate the experimental data
    # n_data_pts = 100
    # Time at the peak (shifted)
    # start_point_time = tt[RF.argmax()]# - shift
    # Disp at the peak (shifted)
    # start_point_disp = dn[RF.argmax()]# - shift_disp 
    # f_exp_time = interpolate.interp1d(dfNew['t'], dfNew['y'])
    # f_exp_disp = interpolate.interp1d(dfNew['x'], dfNew['y'])
    # t_new_exp = np.linspace(start_point_time, tt[tt.argmax()], 
    #                         n_data_pts) # (s)
    # x_new_exp = np.linspace(start_point_disp, dn[tt.argmax()], 
    #                         n_data_pts) # (mm)
    # y_new_exp_time = f_exp_time(t_new_exp) # Interpolate `interp1d`
    # y_new_exp_disp = f_exp_disp(x_new_exp) # Interpolate `interp1d`
    
    # In[Interpolated Simulated Trace]
    
    # # Interpolate the simulated data
    # f_sim_time = interpolate.interp1d(tt, RF)
    # f_sim_disp = interpolate.interp1d(dn, RF)
    # t_new_sim = np.linspace(start_point_time, tt[tt.argmax()], 
    #                         n_data_pts) # (s)
    # x_new_sim = np.linspace(start_point_disp, dn[tt.argmax()], 
    #                         n_data_pts) # (mm)
    # y_new_sim_time = f_sim_time(t_new_sim) # Interpolate `interp1d`
    # y_new_sim_disp = f_sim_disp(x_new_sim) # Interpolate `interp1d`
    
    # In[Plots]
    # ''' Time curve '''
    # fit, ax = plt.subplots()
    # ax.plot()
    # ax.plot(dfdata['t'], dfdata['y'], label='Original Shifted Data', 
    #         alpha = 0.5)
    # ax.plot(dfNew['t'], dfNew['y'], label='Merged Data', 
    #         alpha = 0.5)
    # ax.plot(t_new_exp, y_new_exp_time, '--', 
    #         label='Interp Experimental Data')
    # ax.plot(tt, RF, label='Simulated Data')
    # ax.plot(t_new_sim, y_new_sim_time, ':', label='Interp Simulated Data')
    # ax.set_xlim([0, 300])
    # ax.set_xlabel('Time (s)', fontsize=14)
    # ax.set_ylabel('Force (N)', fontsize=14)
    # ax.legend(loc='best', fontsize=14)
    # ax.grid('on')
    # plt.savefig(os.path.join(abqWD, 'interp1d_Time.pdf'), dpi=300, 
    #             bbox_inches='tight')
    # plt.show()
    
    # ''' Displacement curve '''
    # fit, ax = plt.subplots()
    # ax.plot()
    # ax.plot(dfdata['x'], dfdata['y'], label='Original Shifted Data', 
    #         alpha = 0.5)
    # ax.plot(dfNew['x'], dfNew['y'], label='Merged Data', 
    #         alpha = 0.5)
    # ax.plot(x_new_exp, y_new_exp_disp, '--', 
    #         label='Interp Experimental Data')
    # ax.plot(dn, RF, label='Simulated Data')
    # ax.plot(x_new_sim, y_new_sim_disp, ':', label='Interp Simulated Data')
    # ax.set_xlim([0, max(dn)])
    # ax.set_xlabel('Displacement (mm)', fontsize=14)
    # ax.set_ylabel('Force (N)', fontsize=14)
    # ax.legend(loc='best', fontsize=14)
    # ax.grid('on')
    # plt.savefig(os.path.join(abqWD, 'interp1d_Disp.pdf'), dpi=300, 
    #             bbox_inches='tight')
    # plt.show()
    
    # ''' Displacement curve only showing interpolated data '''
    # abs residual calculation
    # residual = abs(y_new_exp_disp - y_new_sim_disp) 
    # L2Norm = np.dot(residual, residual)
    
    # fit, ax = plt.subplots()
    # ax.plot()
    # ax.plot(x_new_exp, y_new_exp_disp, '-', label='Interp Experimental Data')
    # ax.plot(x_new_sim, y_new_sim_disp, '-', label='Interp Simulated Data')
    # ax.plot(x_new_sim, residual, ':', 
    #         label=r'Residual = $\|\| exp - sim \|\|$', alpha = 0.8)
    # ax.plot([], [], color='white', 
    #         label=r'$L^2$ norm = {:.4f}'.format(L2Norm))
    # ax.axhline(color='k', linewidth=0.25)
    # ax.set_xlim([0, max(x_new_exp)])
    # ax.set_xlabel('Displacement (mm)', fontsize=14)
    # ax.set_ylabel('Force (N)', fontsize=14)
    # ax.legend(loc='best', fontsize=14)
    # ax.grid('on')
    # plt.savefig(os.path.join(abqWD, 'interp1d_Disp_clean.pdf'), dpi=300, 
    #             bbox_inches='tight')
    # plt.show()
    
    print('Output files have been printed to determine the appropriate ' + 
          'parameters for the simulation')
    
    returnArray = [max(fit_t), max(dfn_ss_time_shift), HTMax_shift, 
                   Integral*1e-6]
    return returnArray

if __name__ == '__main__':
    # Run the function
    
    # fileName = sys.argv[-2]
    # savePath = sys.argv[-1]
    
    ReadRAWDataTrace(fileName, abqWD, timeBeforePeak)

