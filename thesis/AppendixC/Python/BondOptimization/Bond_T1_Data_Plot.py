# -*- coding: utf-8 -*-
"""
Created on Fri Apr 03 11:58:40 2020

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

# jobName = sys.argv[1] # Extract the jobName from the previous script

#if __name__ == '__main__':
def plot_Field_Output(fileName, dataDirectory, dataCompare, BondStatus, PDFMStatus):

    """ Field Output Data """
    df = pd.read_csv(os.path.join(dataDirectory,fileName), sep="\t", header=0)
    
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

    t = df.Time
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
    TD = df.Glue_Displacements*1e3 # Convert from m to mm
    BD = df.Bond_Displacements*1e3 # Convert from m to mm
    
    tabArea = 0.00002247 # m^2 (Used in the hand calc - not used anymore)
    
    (figureName, ext) = os.path.splitext(fileName) # Split the file extension
    
    """ Read in the csv file """
    dfValsn = pd.read_csv(dataCompare, sep="\t", nrows=22, header=None, names=['Var','Attribute'])
    
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
    dfn_force = dfn.Force*1e3 # convert from N to mN
    
    # SS Array
    ssTimeArray = np.array([HSSi, HSSf])
    ssValArray = np.array([HFSS, HFSS])
    
    # Max peel force displacement at max and steady state
    dfn_max_Disp = dfn_extension[dfn_time == HTMax]
    dfn_ss_Disp = np.array([dfn_extension[dfn_time == HSSi].values[0], dfn_extension[dfn_time == HSSf].values[0]])
    
    # Plot the data trace to compare the simulated results with the force displacement curves
    plt.plot(dfn_extension, dfn_force,'-', color='r', linewidth=1, markersize=2, label = '{}, Age: {}'.format(HID, HAGE))
    if str(HFMax) == 'nan' and str(HSSi) == 'nan':
        print('No max or steady state')
        pass
    
    if str(HFMax) != 'nan':
        plt.plot(dfn_max_Disp, HFMax,'.', color='k', linewidth=1, markersize=20, label = 'Max Peel - {:.4f} (mN)'.format(HFMax), path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
    
    if str(HSSi) != 'nan':
        plt.plot(dfn_ss_Disp, ssValArray,'-', color='c', linewidth=3, markersize=2, label = 'Steady State - {:.4f} (mN)'.format(HFSS), path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
    
    """ Plots """
    ################## Plot Data #########################
    plt.plot(TD, RF,'-',color='blue',linewidth=2,markersize=2,label = r'Simulated Reaction force $\Sigma F_{Retina}$')
    plt.xlabel('Displacement (mm)',fontsize=18)
    plt.ylabel('Force (mN)',fontsize=18)
    plt.title('Vitreous',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + '_RF_vs_Disp.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    # Plot the data trace to compare the simulated results
    plt.plot(dfn_time, dfn_force,'-', color='r', linewidth=1, markersize=2, label = '{}, Age: {}'.format(HID, HAGE))
    if str(HFMax) == 'nan' and str(HSSi) == 'nan':
        print('No max or steady state')
        pass
    
    if str(HFMax) != 'nan':
        plt.plot(HTMax, HFMax,'.', color='k', linewidth=1, markersize=20, label = 'Max Peel - {:.4f} (mN)'.format(HFMax), path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
    
    if str(HSSi) != 'nan':
        plt.plot(ssTimeArray, ssValArray,'-', color='c', linewidth=3, markersize=2, label = 'Steady State - {:.4f} (mN)'.format(HFSS), path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
    
    """ Plots """
    ################## Plot Data #########################
    plt.plot(t, RF,'-',color='blue',linewidth=2,markersize=2,label = r'Simulated Reaction force $\Sigma F_{Retina}$')
    plt.xlabel('Time (sec)',fontsize=18)
    plt.ylabel('Force (mN)',fontsize=18)
    plt.title('Vitreous',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + '_RF_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    """ Sum Nodal Force Reaction force """
    ################## Plot Data #########################
    plt.plot(t, NF,'-',color='blue',linewidth=2,markersize=2,label = 'Reaction force')
    plt.xlabel('Time (sec)',fontsize=18)
    plt.ylabel('Force (mN)',fontsize=18)
    plt.title('Vitreous',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + '_NForce_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    """ Compare sum RF vs Nforce """
    ################## Plot Data #########################
    plt.plot(t, RF,'-',color='blue',linewidth=2,markersize=2,label = 'RF')
    plt.plot(t, NF,':',color='red',linewidth=2,markersize=2,label = 'NFORC')
    plt.xlabel('Time (sec)',fontsize=18)
    plt.ylabel('Force (mN)',fontsize=18)
    plt.title('Vitreous',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + '_RF_vs_NForce_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    """ Plot bond disp """
    ################## Plot Data #########################
    plt.plot(t, BD,'-',color='blue',linewidth=2,markersize=2,label = 'Bond - Disp')
    plt.plot(t, TD,'-',color='red',linewidth=2,markersize=2,label = 'Top - Disp')
    plt.xlabel('Time (sec)',fontsize=18)
    plt.ylabel('Bond Disp (mm)',fontsize=18)
    plt.title('Vitreous',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + '_disp_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    """ Contact Nforce """
    ################## Plot Data #########################
    plt.plot(t, CnF_RV,'-',color='red',linewidth=2,markersize=2,label = r'CnormF$_{RV}$')
    plt.plot(t, CnF_VR,':',color='blue',linewidth=2,markersize=2,label = r'CnormF$_{VR}$')
    plt.xlabel('Time (sec)',fontsize=18)
    plt.ylabel('Force (mN)',fontsize=18)
    plt.title('Contact Normal Force',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + 'CnormF_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    """ Contact Cpress """
    ################## Plot Data #########################
    plt.plot(t, Cp_RV,'-',color='red',linewidth=2,markersize=2,label = r'Cpress$_{RV}$')
    plt.plot(t, Cp_VR,':',color='blue',linewidth=2,markersize=2,label = r'Cpress$_{VR}$')
    plt.xlabel('Time (sec)',fontsize=18)
    plt.ylabel('Pressure (Pa)',fontsize=18)
    plt.title('Contact pressure',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + 'Cpress_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    """ Contact AVG_Cpress """
    ################## Plot Data #########################
    plt.plot(t, AVG_Cp_RV,'-',color='red',linewidth=2,markersize=2,label = r'AVG Cpress$_{RV}$')
    plt.plot(t, AVG_Cp_VR,':',color='blue',linewidth=2,markersize=2,label = r'AVG Cpress$_{VR}$')
    plt.xlabel('Time (sec)',fontsize=18)
    plt.ylabel('Pressure (Pa)',fontsize=18)
    plt.title('Contact pressure Average',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + 'AVG_Cpress_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    """ Contact Cshear1 """
    ################## Plot Data #########################
    plt.plot(t, Cs1_RV,'-',color='red',linewidth=2,markersize=2,label = r'Cshear$_{RV}^1$')
    plt.plot(t, Cs1_VR,':',color='blue',linewidth=2,markersize=2,label = r'Cshear$_{VR}^1$')
    plt.xlabel('Time (sec)',fontsize=18)
    plt.ylabel('Shear Force (mN)',fontsize=18)
    plt.title('Contact shear 1 force',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + 'Cshear1_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    """ Contact Cshear2 """
    ################## Plot Data #########################
    plt.plot(t, Cs2_RV,'-',color='red',linewidth=2,markersize=2,label = r'Cshear$_{RV}^2$')
    plt.plot(t, Cs2_VR,':',color='blue',linewidth=2,markersize=2,label = r'Cshear$_{VR}^2$')
    plt.xlabel('Time (sec)',fontsize=18)
    plt.ylabel('Shear Force (mN)',fontsize=18)
    plt.title('Contact shear 2 force',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + 'Cshear2_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    """ Contact CshearF """
    ################## Plot Data #########################
    plt.plot(t, CsF_RV,'-',color='red',linewidth=2,markersize=2,label = r'Cshear$_{RV}^F$')
    plt.plot(t, CsF_VR,':',color='blue',linewidth=2,markersize=2,label = r'Cshear$_{VR}^F$')
    plt.xlabel('Time (sec)',fontsize=18)
    plt.ylabel('Shear Force (mN)',fontsize=18)
    plt.title('Contact shear F force',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + 'CshearF_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()

def plot_History_Output(fileName, dataDirectory):
    """ History Output Data """
    df = pd.read_csv(os.path.join(dataDirectory, fileName), sep="\t", header=0)
    df.columns = ["Frame","Time","Internal_Energy","Kinetic_Energy"]
    
    t_h = df.Time
    IE = df.Internal_Energy
    KE = df.Kinetic_Energy
    
    (figureName, ext) = os.path.splitext(fileName) # Split the file extension
    
    """ Plots History Outputs """
    ################## Plot Data #########################
    plt.plot(t_h, IE,'-',color='blue',linewidth=2,markersize=2,label = 'Internal Energy')
    plt.plot(t_h, KE,'-',color='red',linewidth=2,markersize=2,label = 'Kinetic Energy')
    plt.xlabel('Time (sec)',fontsize=18)
    plt.ylabel('Energy (J)',fontsize=18)
    plt.title('Energy',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + '_Energy_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    ################## Plot Data #########################
    plt.semilogy(t_h, KE/IE,'-',color='blue',linewidth=2,markersize=2,label = r'Ratio $\frac{KE}{IE}$')
    plt.semilogy(t_h, 0.1*np.ones(len(t_h)),'-',color='red',linewidth=2,markersize=2,label = '10%')
    plt.xlabel('Time (sec)',fontsize=18)
    plt.ylabel('Ratio of KE to IE',fontsize=18)
    plt.title('Energy ratio',fontsize=20)
    plt.grid()
    plt.legend(loc = 'best',fontsize = 'medium')
    plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + '_Ratio_KE_IE_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    # ################## Plot Data #########################
    # plt.plot(t_h, gRP_RF,'-',color='blue',linewidth=2,markersize=2,label = r'G$_{RP}$')
    # plt.xlabel('Time (sec)',fontsize=18)
    # plt.ylabel('Reaction Force (mN)',fontsize=18)
    # plt.title('Glue Reference Point History Output',fontsize=20)
    # plt.grid()
    # plt.legend(loc = 'best',fontsize = 'medium')
    # plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + '_Glue_RP_RF.png'),dpi=300, bbox_inches='tight') # Save figure
    # plt.close()
    
    # ################## Plot Data #########################
    # plt.plot(t_h, CFNCP_RG*1e3,'-',color='red',linewidth=2,markersize=2,label = r'CFNCP$_{RG}$')
    # plt.plot(t_h, CFNCP_GR*1e3,':',color='blue',linewidth=2,markersize=2,label = r'CFNCP$_{GR}$')
    # plt.xlabel('Time (sec)',fontsize=18)
    # plt.ylabel('Reaction Force (mN)',fontsize=18)
    # plt.title('Contact Force CFNCP_RG/GR History Output',fontsize=20)
    # plt.grid()
    # plt.legend(loc = 'best',fontsize = 'medium')
    # plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + '_CFNCP_RG_GR_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    # plt.close()
    
    # ################## Plot Data #########################
    # plt.plot(t_h, CFNCP_RV*1e3,'-',color='red',linewidth=2,markersize=2,label = r'CFNCP$_{RV}$')
    # plt.plot(t_h, CFNCP_VR*1e3,':',color='blue',linewidth=2,markersize=2,label = r'CFNCP$_{VR}$')
    # plt.xlabel('Time (sec)',fontsize=18)
    # plt.ylabel('Reaction Force (mN)',fontsize=18)
    # plt.title('Contact Force CFNCP_RV/GR History Output',fontsize=20)
    # plt.grid()
    # plt.legend(loc = 'best',fontsize = 'medium')
    # plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + '_CFNCP_RV_VR_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    # plt.close()
    
    # ################## Plot Data #########################
    # plt.plot(t_h, CAreaCP_RG,'-',color='red',linewidth=2,markersize=2,label = r'CAreaCP$_{RG}$')
    # plt.plot(t_h, CAreaCP_GR,':',color='blue',linewidth=2,markersize=2,label = r'CAreaCP$_{GR}$')
    # plt.xlabel('Time (sec)',fontsize=18)
    # plt.ylabel(r'CArea ($m^2$)',fontsize=18)
    # plt.title('Contact Area CAreaCP_RG/GR History Output',fontsize=20)
    # plt.grid()
    # plt.legend(loc = 'best',fontsize = 'medium')
    # plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + 'CAreaCP_RG_GR_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    # plt.close()
    
    # ################## Plot Data #########################
    # plt.plot(t_h, CAreaCP_RV*1e3,'-',color='red',linewidth=2,markersize=2,label = r'CAreaCP$_{RV}$')
    # plt.plot(t_h, CAreaCP_VR*1e3,':',color='blue',linewidth=2,markersize=2,label = r'CAreaCP$_{VR}$')
    # plt.xlabel('Time (sec)',fontsize=18)
    # plt.ylabel(r'CArea ($m^2$)',fontsize=18)
    # plt.title('Contact Area CAreaCP_RV/VR History Output',fontsize=20)
    # plt.grid()
    # plt.legend(loc = 'best',fontsize = 'medium')
    # plt.savefig(os.path.join(dataDirectory,'Figures/' + figureName + 'CAreaCP_RV_VR_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    # plt.close()
    
    print("Plots will be in the figures folder")

def plotFieldHist(FieldfileName, HistfileName, dataDirectory, dataCompare, BondStatus, PDFMStatus):
    """ Field Output Data """
    df1 = pd.read_csv(os.path.join(dataDirectory,FieldfileName), sep="\t", header=0)
    
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
    
    df1.columns = Header

    t = df1.Time
    RF = df1.RF_y_dot*1e3 # Convert from N to mN
    NF = df1.Nodal_Force*1e3 # Convert from N to mN
    CnF_RV = df1.CnormF_RV
    CnF_VR = df1.CnormF_VR
    Cp_RV = df1.Cpress_RV
    Cp_VR = df1.Cpress_VR
    AVG_Cp_RV = df1.AVG_Cpress_RV_AVG
    AVG_Cp_VR = df1.AVG_Cpress_VR_AVG
    Cs1_RV = df1.Cshear1_RV
    Cs1_VR = df1.Cshear1_VR
    Cs2_RV = df1.Cshear2_RV
    Cs2_VR = df1.Cshear2_VR
    CsF_RV = df1.CshearF_RV
    CsF_VR = df1.CshearF_VR
    TD = df1.Glue_Displacements*1e3 # Convert from m to mm
    BD = df1.Bond_Displacements*1e3 # Convert from m to mm
    
    (figureName, ext) = os.path.splitext(FieldfileName) # Split the file extension
    
    """ History Output Data """
    df2 = pd.read_csv(os.path.join(dataDirectory, HistfileName), sep="\t", header=0, index_col=False)
    
    df2.columns = ["Frame","Time","Internal_Energy","Kinetic_Energy"]
    
    t_h = df2.Time
    IE = df2.Internal_Energy
    KE = df2.Kinetic_Energy
    
    (figureName, ext) = os.path.splitext(HistfileName) # Split the file extension

    # ################## Plot Data #########################
    # plt.plot(t, RF,'-',color='blue',linewidth=2,markersize=2,label = r'Field $G^{RP}$ RF')
    # plt.plot(t, NF,'--',color='red',linewidth=2,markersize=2,label = 'Field NFORC R')
    # plt.plot(t_h, gRP_RF,':',color='black',linewidth=2,markersize=2,label = r'Hist $G^{RP}$ RF')
    # plt.xlabel('Time (sec)',fontsize=18)
    # plt.ylabel('Force (mN)',fontsize=18)
    # plt.title('Reaction Force Compare Field to History Outputs',fontsize=20)
    # plt.grid()
    # plt.legend(loc = 'best',fontsize = 'medium')
    # plt.savefig(os.path.join(dataDirectory,'Figures/' + 'CompareFieldtoHist_RF.png'),dpi=300, bbox_inches='tight') # Save figure
    # plt.close()

def plot_BondStat_Output(fileName, dataDirectory):
    """ BondStat Output Data """
    df = pd.read_csv(os.path.join(dataDirectory, fileName), sep="\t", header=0)
    
    t = df['Time (s)']
    
    # determine the length of the number of bonded nodes
    # linspace from 0 to 1 by the number of nodes for the y-position
    # Loop over the number of bonded nodes and plot the y-th position vs time with the color of the bond load on a single plot
    
    fig1, ax1 = plt.subplots()
    nRows  = np.shape(df)[0]
    nCols = np.shape(df)[1] - 1 # subtract the time column
    y = np.linspace(0,1,nCols)
    count = 0
    for (colName, colData) in df.iteritems():
        if colName.find('Time') == -1:
            """ Plots History Outputs """
            ################## Plot Data #########################
            sc = ax1.scatter(t, np.ones(nRows)*y[count], c=colData, cmap=cm.cool, s=5, edgecolors='none', vmin=0, vmax=1)
            count += 1 # update the counter
        else:
            continue
    
    # plt.gray() # turns image to grayscale
    plt.colorbar(sc)
    ax1.set_xlabel('Time (sec)',fontsize=18)
    ax1.set_ylabel('Bonded Nodes',fontsize=18)
    ax1.set_title('BONDSTAT (Color indicates status)',fontsize=20)
    (figureName, ext) = os.path.splitext(fileName) # Split the file extension
    fig1.savefig(os.path.join(dataDirectory,'Figures/' + figureName + '_BONDSTAT_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    print("Plots will be in the figures folder")

def plot_BondLoad_Output(fileName, dataDirectory):
    """ BondLoad Output Data """
    df = pd.read_csv(os.path.join(dataDirectory, fileName), sep="\t", header=0)
    
    t = df['Time (s)']
    
    # determine the length of the number of bonded nodes
    # linspace from 0 to 1 by the number of nodes for the y-position
    # Loop over the number of bonded nodes and plot the y-th position vs time with the color of the bond load on a single plot
    
    fig1, ax1 = plt.subplots()
    nRows  = np.shape(df)[0]
    nCols = np.shape(df)[1] - 1 # subtract the time column
    y = np.linspace(0,1,nCols)
    count = 0
    for (colName, colData) in df.iteritems():
        if colName.find('Time') == -1:
            """ Plots History Outputs """
            ################## Plot Data #########################
            sc = ax1.scatter(t, np.ones(nRows)*y[count], c=colData, cmap=cm.cool, s=5, edgecolors='none', vmin=0, vmax=1)
            count += 1 # update the counter
        else:
            continue
    
    # plt.gray() # turns image to grayscale
    plt.colorbar(sc)
    ax1.set_xlabel('Time (sec)',fontsize=18)
    ax1.set_ylabel('Bonded Nodes',fontsize=18)
    ax1.set_title('BONDLOAD (Color indicates status)',fontsize=20)
    (figureName, ext) = os.path.splitext(fileName) # Split the file extension
    fig1.savefig(os.path.join(dataDirectory,'Figures/' + figureName + '_BONDLOAD_vs_t.png'),dpi=300, bbox_inches='tight') # Save figure
    plt.close()
    
    print("Plots will be in the figures folder")

def PlotAbqData(fileName, dataDirectory, dataCompare, BondStatus, PDFMStatus):
    
    # """ Change directory to correct path """
    # filePath = os.getcwd()
    # data_directory = os.path.join(filePath,jobName)
    # figures_directory = os.path.join(filePath,jobName,'Figures')
    # if not os.path.exists(figures_directory):
    #     os.makedirs(figures_directory)
    
    """ Call both functions to plot Field/History data """
    field_files = [f for f in os.listdir(dataDirectory) if os.path.isfile(os.path.join(dataDirectory, f)) and f.startswith('output_Field')]
    for fname in field_files:
        plot_Field_Output(fname, dataDirectory, dataCompare, BondStatus, PDFMStatus)
        
    history_files = [f for f in os.listdir(dataDirectory) if os.path.isfile(os.path.join(dataDirectory, f)) and f.startswith('output_History')]
    for hname in history_files:
        plot_History_Output(hname, dataDirectory)
    
    for fname in field_files:
        plotFieldHist(fname, hname, dataDirectory, dataCompare, BondStatus, PDFMStatus)

    if BondStatus == True:
        BONDSTAT_files = [f for f in os.listdir(dataDirectory) if os.path.isfile(os.path.join(dataDirectory, f)) and f.startswith('BONDSTAT')]
        for bname in BONDSTAT_files:
            plot_BondStat_Output(bname, dataDirectory)
        
        BONDLOAD_files = [f for f in os.listdir(dataDirectory) if os.path.isfile(os.path.join(dataDirectory, f)) and f.startswith('BONDLOAD')]
        for bname in BONDLOAD_files:
            plot_BondLoad_Output(bname, dataDirectory)
