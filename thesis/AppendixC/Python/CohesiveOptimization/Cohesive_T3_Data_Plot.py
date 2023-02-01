# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 23:49:03 2021

@author: Kiffer Creveling
python3

"""
# Packages & path folder
#from sys import argv, exit
#sys.path.append(r'F:\Abaqus Working Directory')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patheffects as pe
import numpy as np
import os
import os.path
import sys
import pdb
plt.rcParams['figure.figsize'] = [16, 9]

def plot_Field_Output(fileName, dataDirectory, dataCompare, 
                      DMGInitiation, DMGEvolution):
    
    """ Field Output Data """
    df = pd.read_csv(os.path.join(dataDirectory, fileName), 
                     sep="\t", header=0)
    
    Header = [] # Header information for the dataframe
    Header.append('Frame')
    Header.append('Time')
    Header.append('RF_y_dot')
    Header.append('RFx')
    Header.append('RFy')
    Header.append('RFz')
    Header.append('Nodal_Force')
    Header.append('Tab_Displacement')
    Header.append('Bond_Displacement')
    Header.append('Stress')
    Header.append('AVG_CSMAXSCRT')
    Header.append('AVG_CSDMG')
    
    df.columns = Header
    
    t = df.Time
    RF = df.RF_y_dot*1e3 # Convert from N to mN
    NF = df.Nodal_Force*1e3 # Convert from N to mN
    TD = df.Tab_Displacement*1e3 # convert from N to mN
    B = df.Bond_Displacement*1e3 # convert from N to mN
    S = df.Stress
    AVG_CSMAXSCRT = df.AVG_CSMAXSCRT
    AVG_CSDMG = df.AVG_CSDMG
    
    (figureName, ext) = os.path.splitext(fileName) # Split the file extension
    
    """ Read in the csv file """
    dfValsn = pd.read_csv(dataCompare, sep="\t", nrows=22, header=None, 
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
    # (mN/m) slope from 20 seconds prior to max force value
    HSlope20 = float(dfValsn['Attribute'][20])
    
    dfn = pd.read_csv(os.path.join(dataCompare), sep="\t", header=30)
    dfn.columns = ['Time', 'Extension', 'Force']
    dfn_time = dfn.Time
    dfn_extension = dfn.Extension
    dfn_force = dfn.Force*1e3 # convert from N to mN
    
    # SS Array
    ssTimeArray = np.array([HSSi, HSSf])
    ssValArray = np.array([HFSS, HFSS])
    
    # Max peel force displacement at max and steady state
    dfn_max_Disp = dfn_extension[dfn_time == HTMax]
    # .flatten()
    dfn_ss_Disp = np.array([dfn_extension[dfn_time == HSSi].values[0], 
                            dfn_extension[dfn_time == HSSf].values[0]])
    
    # Plot the data trace to compare the simulated results with the force 
    # displacement curves
    plt.plot(dfn_extension, dfn_force, '-', color='r', linewidth=1, 
             markersize=2, label = '{}, Age: {}'.format(HID, HAGE))
    if str(HFMax) == 'nan' and str(HSSi) == 'nan':
        print('No max or steady state')
        pass
    
    if str(HFMax) != 'nan':
        plt.plot(dfn_max_Disp, HFMax, '.', color='k', linewidth=1, 
                 markersize=20, label='Max Peel - {:.4f} (mN)'.format(HFMax), 
                 path_effects=[pe.Stroke(linewidth=4, foreground='k'), 
                               pe.Normal()])
    
    if str(HSSi) != 'nan':
        plt.plot(dfn_ss_Disp, ssValArray, '-', color='c', linewidth=3, 
                 markersize=2, label='Steady State - {:.4f} (mN)'.format(HFSS), 
                 path_effects=[pe.Stroke(linewidth=5, foreground='k'), 
                               pe.Normal()])
    
    """ Plots """
    ################## Plot Data #########################
    plt.plot(TD, RF, '-', color='blue', linewidth=2, markersize=2, 
             label = r'Simulated Reaction force $\Sigma F_{Retina}$')
    plt.xlabel('Displacement (mm)', fontsize=18)
    plt.ylabel('Force (mN)', fontsize=18)
    plt.title('Vitreous', fontsize=20)
    plt.grid()
    plt.legend(loc = 'best', fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory, 'Figures/' + 
                             figureName + '_RF_vs_Disp.pdf'), 
                dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    # Plot the data trace to compare the simulated results
    plt.plot(dfn_time, dfn_force, '-', color='r', linewidth=1, markersize=2, 
             label = '{}, Age: {}'.format(HID, HAGE))
    if str(HFMax) == 'nan' and str(HSSi) == 'nan':
        print('No max or steady state')
        pass
    
    if str(HFMax) != 'nan':
        plt.plot(HTMax, HFMax, '.', color='k', linewidth=1, markersize=20, 
                 label = 'Max Peel - {:.4f} (mN)'.format(HFMax), 
                 path_effects=[pe.Stroke(linewidth=4, foreground='k'), 
                               pe.Normal()])
    
    if str(HSSi) != 'nan':
        plt.plot(ssTimeArray, ssValArray, '-', color='c', linewidth=3, 
                 markersize=2, label='Steady State - {:.4f} (mN)'.format(HFSS), 
                 path_effects=[pe.Stroke(linewidth=5, foreground='k'), 
                               pe.Normal()])
    
    """ Plots """
    ################## Plot Data #########################
    plt.plot(t, RF, '-', color='blue', linewidth=2, markersize=2, 
             label = r'Simulated Reaction force $\Sigma F_{Retina}$')
    plt.xlabel('Time (sec)', fontsize=18)
    plt.ylabel('Force (mN)', fontsize=18)
    plt.title('Vitreous', fontsize=20)
    plt.grid()
    plt.legend(loc = 'best', fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory, 'Figures/' + 
                             figureName + '_RF_vs_t.pdf'), 
                dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    ################## Plot Data #########################
    plt.plot(t, NF, '-', color='blue', linewidth=2, markersize=2, 
             label = 'Reaction force NForce')
    plt.xlabel('Time (sec)', fontsize=18)
    plt.ylabel('Force (N)', fontsize=18)
    plt.title('Vitreous', fontsize=20)
    plt.grid()
    plt.legend(loc = 'best', fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory, 'Figures/' + 
                             figureName + '_NF_vs_t.pdf'), 
                dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    ################## Plot Data #########################
    plt.plot(t, B, '-', color='blue', linewidth=2, markersize=2, 
             label = 'Bond - Disp')
    plt.xlabel('Time (sec)', fontsize=18)
    plt.ylabel('Bond Disp (mm)', fontsize=18)
    plt.title('VR Interface', fontsize=20)
    plt.grid()
    plt.legend(loc = 'best', fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory, 'Figures/' + 
                             figureName + '_B_vs_t.pdf'), 
                dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    ################## Plot Data #########################
    plt.plot(t, B,'-', color='blue', linewidth=2, markersize=2, 
             label = 'Bond - Disp')
    plt.plot(t, TD, '-.', color='red', linewidth=2, markersize=2, 
             label = 'Top - Disp')
    plt.xlabel('Time (sec)', fontsize=18)
    plt.ylabel('Bond Disp (mm)', fontsize=18)
    plt.title('Vitreous', fontsize=20)
    plt.grid()
    plt.legend(loc = 'best', fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory, 'Figures/' + 
                             figureName + '_disp_vs_t.pdf'), 
                dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    ################## Plot Data #########################
    plt.plot(t, S, '-', color='blue', linewidth=2, markersize=2, 
             label = 'Stress')
    plt.xlabel('Time (sec)', fontsize=18)
    plt.ylabel('Stress (Pa)', fontsize=18)
    plt.title('Vitreous', fontsize=20)
    plt.grid()
    plt.legend(loc = 'best', fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory, 'Figures/' + 
                             figureName + '_Stress_vs_t.pdf'), 
                dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    if DMGInitiation == True:
        ################## Plot Data #########################
        plt.plot(t, AVG_CSMAXSCRT, '-', color='blue', linewidth=2, 
                 markersize=2, label = r'CSMAXSCRT$_{AVG}$')
        plt.xlabel('Time (sec)', fontsize=18)
        plt.ylabel('Maximum Displacement Criterion Value', fontsize=18)
        plt.title('Vitreous', fontsize=20)
        plt.grid()
        plt.legend(loc = 'best', fontsize = 'medium')
        plt.savefig(os.path.join(dataDirectory, 'Figures/' + 
                                 figureName + '_AVG_CSMAXSCRT_vs_t.pdf'), 
                    dpi=300, bbox_inches='tight') # Save figure
        plt.close()
    
    if DMGEvolution == True:
        ################## Plot Data #########################
        plt.plot(t, AVG_CSDMG, '-', color='blue', linewidth=2, 
                 markersize=2, label = r'CSDMG$_{AVG}$')
        plt.xlabel('Time (sec)', fontsize=18)
        plt.ylabel('Maximum Damage Value', fontsize=18)
        plt.title('Vitreous', fontsize=20)
        plt.grid()
        plt.legend(loc = 'best', fontsize = 'medium')
        plt.savefig(os.path.join(dataDirectory, 'Figures/' + 
                                 figureName + '_AVG_CSDMG_vs_t.pdf'), 
                    dpi=300, bbox_inches='tight') # Save figure
        plt.close()


def plot_History_Output(fileName, dataDirectory):
    """ History Output Data """
    df = pd.read_csv(os.path.join(dataDirectory, fileName), sep="\t", header=0)
    df.columns = ["Frame", "Time", "Internal_Energy", "Kinetic_Energy"]
    
    t_h = df.Time
    IE = df.Internal_Energy
    KE = df.Kinetic_Energy
    
    (figureName, ext) = os.path.splitext(fileName) # Split the file extension
    
    """ Plots History Outputs """
    ################## Plot Data #########################
    plt.plot(t_h, IE, '-', color='blue', linewidth=2, markersize=2, 
             label = 'Internal Energy')
    plt.plot(t_h, KE, '-', color='red', linewidth=2, markersize=2, 
             label = 'Kinetic Energy')
    plt.xlabel('Time (sec)', fontsize=18)
    plt.ylabel('Energy (J)', fontsize=18)
    plt.title('Energy', fontsize=20)
    plt.grid()
    plt.legend(loc = 'best', fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory, 'Figures/' + 
                             figureName + '_Energy.pdf'), 
                dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    ################## Plot Data #########################
    plt.semilogy(t_h, KE/IE, '-', color='blue', linewidth=2, 
                 markersize=2, label = r'Ratio $\frac{KE}{IE}$')
    plt.semilogy(t_h, 0.1*np.ones(len(t_h)), '-', color='red', 
                 linewidth=2, markersize=2, label = '10%')
    plt.xlabel('Time (sec)', fontsize=18)
    plt.ylabel('Ratio of KE to IE', fontsize=18)
    plt.title('Energy ratio', fontsize=20)
    plt.grid()
    plt.legend(loc = 'best', fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory, 'Figures/' + 
                             figureName + '_Ratio_KE_IE.pdf'), 
                dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    print("Plots will be in the figures folder")

def plot_CohesiveCSMAXSCRT_Output(fileName, dataDirectory):
    """ CohesiveCSMAXSCRT Output Data """
    df = pd.read_csv(os.path.join(dataDirectory, fileName), sep="\t", header=0)
    
    t = df['Time (s)']
    
    """
    The incoming data has both the Retina and Vitreous nodes associated 
    with it.  We need to split them apart and create plots for each data 
    set separately
    """
    
    # Filter data by the "name" of the node that begins with 1 i.e. '1000002'
    # and create a new dataframe
    dfR = df.loc[:, df.columns.str.startswith('4')] # Retina
    dfV = df.loc[:, df.columns.str.startswith('5')] # Vitreous
    
    """ Retina """
    
    # determine the length of the number of bonded nodes
    # linspace from 0 to 1 by the number of nodes for the y-position
    # Loop over the number of bonded nodes and plot the y-th 
    # vs time with the color of the bond load on a single plot
    
    fig1, ax1 = plt.subplots()
    nRows  = np.shape(dfR)[0]
    nCols = np.shape(dfR)[1]
    y = np.linspace(0, 1, nCols)
    count = 0
    for (colName, colData) in dfR.iteritems():
        if colName.find('Time') == -1:
            """ Plots CSMAXSCRT Outputs """
            ################## Plot Data #########################
            sc = ax1.scatter(t, np.ones(nRows)*y[count], c=colData, 
                             cmap=cm.cool, s=5, edgecolors='none', 
                             vmin=0, vmax=1)
            count += 1 # update the counter
        else:
            continue
    
    # plt.gray() # turns image to grayscale
    plt.colorbar(sc)
    ax1.set_xlabel('Time (sec)', fontsize=18)
    ax1.set_ylabel('Cohesive CSMAXSCRT', fontsize=18)
    ax1.set_title('Retina CSMAXSCRT (Color indicates status)', fontsize=20)
    (figureName, ext) = os.path.splitext(fileName) # Split the file extension
    fig1.savefig(os.path.join(dataDirectory, 'Figures/' + 
                              figureName + '_CSMAXSCRT_vs_t_Retina.pdf'), 
                 dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    """ Vitreous """
    
    # determine the length of the number of bonded nodes
    # linspace from 0 to 1 by the number of nodes for the y-position
    # Loop over the number of bonded nodes and plot the y-th 
    # position vs time with the color of the bond load on a single plot
    
    fig1, ax1 = plt.subplots()
    nRows  = np.shape(dfV)[0]
    nCols = np.shape(dfV)[1] #  - 1 # subtract the time column
    y = np.linspace(0, 1, nCols)
    count = 0
    for (colName, colData) in dfV.iteritems():
        if colName.find('Time') == -1:
            """ Plots CSMAXSCRT Outputs """
            ################## Plot Data #########################
            sc = ax1.scatter(t, np.ones(nRows)*y[count], c=colData, 
                             cmap=cm.cool, s=5, edgecolors='none', 
                             vmin=0, vmax=1)
            count += 1 # update the counter
        else:
            continue
    
    # plt.gray() # turns image to grayscale
    plt.colorbar(sc)
    ax1.set_xlabel('Time (sec)', fontsize=18)
    ax1.set_ylabel('Cohesive CSMAXSCRT', fontsize=18)
    ax1.set_title('Vitreous CSMAXSCRT (Color indicates status)', fontsize=20)
    (figureName, ext) = os.path.splitext(fileName) # Split the file extension
    fig1.savefig(os.path.join(dataDirectory, 'Figures/' + 
                              figureName + '_CSMAXSCRT_vs_t_Vitreous.pdf'), 
                 dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    print("Plots will be in the figures folder")

def plot_CohesiveCSDMG_Output(fileName, dataDirectory):
    """ CohesiveCSDMG Output Data """
    df = pd.read_csv(os.path.join(dataDirectory, fileName), sep="\t", header=0)
    
    t = df['Time (s)']
    
    """
    The incoming data has both the Retina and Vitreous nodes associated 
    with it.  We need to split them apart and create plots for each data 
    set separately
    """
    
    # Filter data by the "name" of the node that begins with 1 i.e. '1000002'
    dfR = df.loc[:, df.columns.str.startswith('4')] # Retina
    dfV = df.loc[:, df.columns.str.startswith('5')] # Vitreous
    
    """ Retina """
    
    # determine the length of the number of bonded nodes
    # linspace from 0 to 1 by the number of nodes for the y-position
    # Loop over the number of bonded nodes and plot the y-th 
    # position vs time with the color of the bond load on a single plot
    
    fig1, ax1 = plt.subplots()
    nRows  = np.shape(dfR)[0]
    nCols = np.shape(dfR)[1] #  - 1 # subtract the time column
    y = np.linspace(0, 1, nCols)
    count = 0
    for (colName, colData) in dfR.iteritems():
        if colName.find('Time') == -1:
            """ Plots CohesiveCSDMG Outputs """
            ################## Plot Data #########################
            sc = ax1.scatter(t, np.ones(nRows)*y[count], c=colData, 
                             cmap=cm.cool, s=5, edgecolors='none', 
                             vmin=0, vmax=1)
            count += 1 # update the counter
        else:
            continue
    
    # plt.gray() # turns image to grayscale
    plt.colorbar(sc)
    ax1.set_xlabel('Time (sec)', fontsize=18)
    ax1.set_ylabel('Cohesive CSDMG', fontsize=18)
    ax1.set_title('Retina CSDMG (Color indicates status)', fontsize=20)
    (figureName, ext) = os.path.splitext(fileName) # Split the file extension
    fig1.savefig(os.path.join(dataDirectory, 'Figures/' + 
                              figureName + '_CSDMG_vs_t_Retina.pdf'), 
                 dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    """ Vitreous """
    
    # determine the length of the number of bonded nodes
    # linspace from 0 to 1 by the number of nodes for the y-position
    # Loop over the number of bonded nodes and plot the y-th 
    # position vs time with the color of the bond load on a single plot
    
    fig1, ax1 = plt.subplots()
    nRows  = np.shape(dfV)[0]
    nCols = np.shape(dfV)[1] #  - 1 # subtract the time column
    y = np.linspace(0, 1, nCols)
    count = 0
    for (colName, colData) in dfV.iteritems():
        if colName.find('Time') == -1:
            """ Plots CohesiveCSDMG Outputs """
            ################## Plot Data #########################
            sc = ax1.scatter(t, np.ones(nRows)*y[count], c=colData, 
                             cmap=cm.cool, s=5, edgecolors='none', 
                             vmin=0, vmax=1)
            count += 1 # update the counter
        else:
            continue
    
    # plt.gray() # turns image to grayscale
    plt.colorbar(sc)
    ax1.set_xlabel('Time (sec)', fontsize=18)
    ax1.set_ylabel('Cohesive CSDMG', fontsize=18)
    ax1.set_title('Vitreous CSDMG (Color indicates status)', fontsize=20)
    (figureName, ext) = os.path.splitext(fileName) # Split the file extension
    fig1.savefig(os.path.join(dataDirectory, 'Figures/' + 
                              figureName + '_CSDMG_vs_t_Vitreous.pdf'), 
                 dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    print("Plots will be in the figures folder")
    
def PlotAbqData(fileName, dataDirectory, dataCompare, 
                DMGInitiation, DMGEvolution):
    
    # """ Change directory to correct path """
    # filePath = os.getcwd()
    # data_directory = os.path.join(filePath, jobName)
    # figures_directory = os.path.join(filePath, jobName, 'Figures')
    # if not os.path.exists(figures_directory):
    #     os.makedirs(figures_directory)

    """ Call both functions to plot Field/History data """
    field_files = [f for f in os.listdir(dataDirectory) 
                   if os.path.isfile(os.path.join(dataDirectory, f)) 
                   and f.startswith('output_Field')]
    for fname in field_files:
        plot_Field_Output(fname, dataDirectory, dataCompare, 
                          DMGInitiation, DMGEvolution)
    
    history_files = [f for f in os.listdir(dataDirectory) 
                     if os.path.isfile(os.path.join(dataDirectory, f)) 
                     and f.startswith('output_History')]
    for hname in history_files:
        plot_History_Output(hname, dataDirectory)
    
    if DMGInitiation == True:
        CSMAXSCRT_files = [f for f in os.listdir(dataDirectory) 
                           if os.path.isfile(os.path.join(dataDirectory, f)) 
                           and f.startswith('CSMAXSCRT')]
        for CSMAXSCRTname in CSMAXSCRT_files:
            plot_CohesiveCSMAXSCRT_Output(CSMAXSCRTname, dataDirectory)
    
    if DMGEvolution == True:
        CSDMG_files = [f for f in os.listdir(dataDirectory) 
                       if os.path.isfile(os.path.join(dataDirectory, f)) 
                       and f.startswith('CSDMG')]
        for CSDMGname in CSDMG_files:
            plot_CohesiveCSDMG_Output(CSDMGname, dataDirectory)