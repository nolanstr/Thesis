# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:45:38 2020

@author: Kiffer2
"""

import pandas as pd
import sqlite3
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm
import matplotlib.patheffects as pe
plt.rcParams['figure.figsize'] = [16, 9]
import pdb


def Least_Squares(x,y):
    """
    Calculate the slope and y-intercept using matrix math
    x & y are the coordinates of points
    
    parameters (X,Y) Data
    
    Returns:
        Curve fit data and parameters m*x + b
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

os.chdir('F:/Abaqus Working Directory/PeterComp/HumanData')

filePath = os.getcwd()

ExcelName = 'HumanData.xlsx'

ExcelPath = os.path.join(filePath, ExcelName)

conn = sqlite3.connect('HumanData.db')
c = conn.cursor()

c.execute('DROP TABLE IF EXISTS HumanData')
c.execute('''CREATE TABLE 'HumanData'(
    'HumanID' TEXT,
    'HumanAGE' REAL,
    'HumanGender' TEXT,
    'HumanLeftRight' TEXT,
    'HumanRegion' TEXT,
    'PostMortemHrs_Min' REAL,
    'DateOfDeath' timestamp,
    'TimeOfDeath' timestamp,
    'EnucleationDate' timestamp,
    'EnucleationTime' timestamp,
    'DateOfTesting' timestamp,
    'TimeOfTesting' timestamp,
    'DiameterPostAnt' REAl,
    'DiameterNasTemp' REAL,
    'SSi' REAL,
    'SSf' REAL,
    'TFMax' REAL,
    'DispMax' REAL,
    'FMax' REAL,
    'FSS' REAL,
    'Slope10' REAL,
    'Rsqrd10' REAL,
    'Slope20' REAL,
    'Rsqrd20' REAL,
    'Slope30' REAL,
    'Rsqrd30' REAL,
    'Slope0' REAL,
    'Rsqrd0' REAL,
    'PeelVideoName' TEXT,
    'PeelVideoHyperlink' TEXT,
    'VideoComments' TEXT,
    'LightMicroscopyImages' TEXT)
''')

df = pd.read_excel(ExcelPath, sheet_name=None)

""" Put values into a dictionary of dataframes """
HumanID = {}
HumanAge = {}
HumanGender = {}
HumanLeftRight = {}
HumanRegion = {}
PostMortemHrs_Min = {}
DateOfDeath = {}
TimeOfDeath = {}
EnucleationDate = {}
EnucleationTime = {}
DateOfTesting = {}
TimeOfTesting = {}
DiameterPostAnt = {}
DiameterNasTemp = {}
SSi = {}
SSf = {}
FMax = {}
FSS = {}
TFMax = {}
DispMax = {}
Slope10 = {}
Rsqrd10 = {}
Slope20 = {}
Rsqrd20 = {}
Slope30 = {}
Rsqrd30 = {}
Slope0 = {}
Rsqrd0 = {}
PeelVideoName = {}
PeelVideoHyperlink = {}
VideoComments = {}
LightMicroscopyImages = {}
time = {}
extension = {}
force = {}
for i in df.keys():
    if len(i) <= 2: # Only look at data traces ... [Row,Col]
        print(i)
        HumanID['{}'.format(i)] =  df['{}'.format(i)].iloc[0,1]
        HumanAge['{}'.format(i)] =  df['{}'.format(i)].iloc[1,1]
        HumanGender['{}'.format(i)] =  df['{}'.format(i)].iloc[2,1]
        HumanLeftRight['{}'.format(i)] =  df['{}'.format(i)].iloc[3,1]
        HumanRegion['{}'.format(i)] =  df['{}'.format(i)].iloc[4,1]
        PostMortemHrs_Min['{}'.format(i)] =  df['{}'.format(i)].iloc[5,1]
        DateOfDeath['{}'.format(i)] =  df['{}'.format(i)].iloc[6,1]
        TimeOfDeath['{}'.format(i)] =  df['{}'.format(i)].iloc[7,1]
        EnucleationDate['{}'.format(i)] =  df['{}'.format(i)].iloc[8,1]
        EnucleationTime['{}'.format(i)] =  df['{}'.format(i)].iloc[9,1]
        DateOfTesting['{}'.format(i)] =  df['{}'.format(i)].iloc[10,1]
        TimeOfTesting['{}'.format(i)] =  df['{}'.format(i)].iloc[11,1]
        DiameterPostAnt['{}'.format(i)] =  df['{}'.format(i)].iloc[12,1]
        DiameterNasTemp['{}'.format(i)] =  df['{}'.format(i)].iloc[13,1]
        SSi['{}'.format(i)] =  df['{}'.format(i)].iloc[18,6]
        SSf['{}'.format(i)] =  df['{}'.format(i)].iloc[19,6]
        FMax['{}'.format(i)] =  df['{}'.format(i)].iloc[22,6]
        FSS['{}'.format(i)] =  df['{}'.format(i)].iloc[21,6]
        PeelVideoName['{}'.format(i)] =  df['{}'.format(i)].iloc[26,6]
        PeelVideoHyperlink['{}'.format(i)] =  df['{}'.format(i)].iloc[27,6]
        VideoComments['{}'.format(i)] =  df['{}'.format(i)].iloc[28,6]
        LightMicroscopyImages['{}'.format(i)] =  df['{}'.format(i)].iloc[29,6]
        
        """ Data Traces """
        time['{}'.format(i)] = pd.to_numeric(df['{}'.format(i)].iloc[17:-1,0].reset_index(drop=True))
        extension['{}'.format(i)] =  pd.to_numeric(df['{}'.format(i)].iloc[17:-1,1].reset_index(drop=True))
        force['{}'.format(i)] =  pd.to_numeric(df['{}'.format(i)].iloc[17:-1,2].reset_index(drop=True))
        
        if str(FMax['{}'.format(i)]) != 'nan':
            # Time at max force
            TFMaxi = time['{}'.format(i)][force['{}'.format(i)].loc[lambda x: x == FMax['{}'.format(i)]].index.values[0]]
            TFMax['{}'.format(i)] = TFMaxi
            
            # Slope calculation 10 seconds before max peel force
            maxIndex = force['{}'.format(i)].loc[lambda x: x == FMax['{}'.format(i)]].index.values[0] # Location in the array for the max force
            x = extension['{}'.format(i)][maxIndex-100:maxIndex] # Array from MaxIndex-100 (10 sec) to location of max force
            y = force['{}'.format(i)][maxIndex-100:maxIndex] # Array from MaxIndex-100 (10 sec) to location of max force
            curveFit, Params, Rsqrd = Least_Squares(x,y) # Perform least squares and return
            Slope10['{}'.format(i)] = Params[1]
            Rsqrd10['{}'.format(i)] = Rsqrd
            
            # Slope calculation 20 seconds before max peel force
            x = extension['{}'.format(i)][maxIndex-200:maxIndex] # Array from MaxIndex-200 (20 sec) to location of max force
            y = force['{}'.format(i)][maxIndex-200:maxIndex] # Array from MaxIndex-200 (20 sec) to location of max force
            curveFit, Params, Rsqrd = Least_Squares(x,y) # Perform least squares and return
            Slope20['{}'.format(i)] = Params[1]
            Rsqrd20['{}'.format(i)] = Rsqrd
            
            # Slope calculation 30 seconds before max peel force
            x = extension['{}'.format(i)][maxIndex-300:maxIndex] # Array from MaxIndex-300 (30 sec) to location of max force
            y = force['{}'.format(i)][maxIndex-300:maxIndex] # Array from MaxIndex-300 (30 sec) to location of max force
            curveFit, Params, Rsqrd = Least_Squares(x,y) # Perform least squares and return
            Slope30['{}'.format(i)] = Params[1]
            Rsqrd30['{}'.format(i)] = Rsqrd
            
            # Slope calculation from zero to max peel force
            x = extension['{}'.format(i)][0:maxIndex] # Array from 0 to location of max force
            y = force['{}'.format(i)][0:maxIndex] # Array from 0 to location of max force
            curveFit, Params, Rsqrd = Least_Squares(x,y) # Perform least squares and return
            Slope0['{}'.format(i)] = Params[1]
            Rsqrd0['{}'.format(i)] = Rsqrd
            
            # Displacement at max force
            DispMax['{}'.format(i)] = extension['{}'.format(i)][time['{}'.format(i)] == TFMaxi].values[0]
            
        else:
            TFMaxi = np.nan
            TFMax['{}'.format(i)] = TFMaxi
            Slope10['{}'.format(i)] = np.nan
            Rsqrd10['{}'.format(i)] = np.nan
            Slope20['{}'.format(i)] = np.nan
            Rsqrd20['{}'.format(i)] = np.nan
            Slope30['{}'.format(i)] = np.nan
            Rsqrd30['{}'.format(i)] = np.nan
            Slope0['{}'.format(i)] = np.nan
            Rsqrd0['{}'.format(i)] = np.nan
            DispMax['{}'.format(i)] = np.nan
        
        """ Add data to SQL database """
        HumanIDi = HumanID['{}'.format(i)]
        HumanAgei = HumanAge['{}'.format(i)]
        HumanGenderi = HumanGender['{}'.format(i)]
        HumanLeftRighti = HumanLeftRight['{}'.format(i)]
        HumanRegioni = HumanRegion['{}'.format(i)]
        PostMortemHrs_Mini = PostMortemHrs_Min['{}'.format(i)]
        DateOfDeathi = DateOfDeath['{}'.format(i)]
        TimeOfDeathi = TimeOfDeath['{}'.format(i)]
        EnucleationDatei = EnucleationDate['{}'.format(i)]
        EnucleationTimei = EnucleationTime['{}'.format(i)]
        DateOfTestingi = DateOfTesting['{}'.format(i)]
        TimeOfTestingi = TimeOfTesting['{}'.format(i)]
        DiameterPostAnti = DiameterPostAnt['{}'.format(i)]
        DiameterNasTempi = DiameterNasTemp['{}'.format(i)]
        SSii = SSi['{}'.format(i)]
        SSfi = SSf['{}'.format(i)]
        TFMaxi = TFMax['{}'.format(i)]
        DispMaxi = DispMax['{}'.format(i)]
        FMaxi = FMax['{}'.format(i)]
        FSSi = FSS['{}'.format(i)]
        Slope10i = Slope10['{}'.format(i)]
        Rsqrd10i = Rsqrd10['{}'.format(i)]
        Slope20i = Slope20['{}'.format(i)]
        Rsqrd20i = Rsqrd20['{}'.format(i)]
        Slope30i = Slope30['{}'.format(i)]
        Rsqrd30i = Rsqrd30['{}'.format(i)]
        Slope0i = Slope0['{}'.format(i)]
        Rsqrd0i = Rsqrd0['{}'.format(i)]
        PeelVideoNamei = PeelVideoName['{}'.format(i)]
        PeelVideoHyperlinki = PeelVideoHyperlink['{}'.format(i)]
        VideoCommentsi = VideoComments['{}'.format(i)]
        LightMicroscopyImagesi = LightMicroscopyImages['{}'.format(i)]
        
        # add to sql database
        # cur.execute(''' INSERT INTO HumanData(HumanID) VALUES (?)''',(HumanIDi))
        # conn.commit()
        c.execute(''' INSERT INTO HumanData(
                        HumanID,
                        HumanAge,
                        HumanGender,
                        HumanLeftRight,
                        HumanRegion,
                        PostMortemHrs_Min,
                        DateOfDeath,
                        TimeOfDeath,
                        EnucleationDate,
                        EnucleationTime,
                        DateOfTesting,
                        TimeOfTesting,
                        DiameterPostAnt,
                        DiameterNasTemp,
                        SSi,
                        SSf,
                        TFMax,
                        DispMax,
                        FMax,
                        FSS,
                        Slope10,
                        Rsqrd10,
                        Slope20,
                        Rsqrd20,
                        Slope30,
                        Rsqrd30,
                        Slope0,
                        Rsqrd0,
                        PeelVideoName,
                        PeelVideoHyperlink,
                        VideoComments,
                        LightMicroscopyImages
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ''',(HumanIDi,
                        HumanAgei,
                        HumanGenderi,
                        HumanLeftRighti,
                        HumanRegioni,
                        str(PostMortemHrs_Mini),
                        str(DateOfDeathi),
                        str(TimeOfDeathi),
                        str(EnucleationDatei),
                        str(EnucleationTimei),
                        str(DateOfTestingi),
                        str(TimeOfTestingi),
                        DiameterPostAnti,
                        DiameterNasTempi,
                        SSii,
                        SSfi,
                        TFMaxi,
                        DispMaxi,
                        FMaxi,
                        FSSi,
                        Slope10i,
                        Rsqrd10i,
                        Slope20i,
                        Rsqrd20i,
                        Slope30i,
                        Rsqrd30i,
                        Slope0i,
                        Rsqrd0i,
                        PeelVideoNamei,
                        PeelVideoHyperlinki,
                        VideoCommentsi,
                        LightMicroscopyImagesi))
        conn.commit()
        

# In[1] Plot individual data traces in the Figures folder as well as the 
# age group/region folder

# Make the new folder to put figure
folderPath = os.path.join(filePath, 'Figures')
if not os.path.exists(folderPath):
    os.makedirs(folderPath)

""" Age Groups """
g1e = [] # 30 - 39 yrs Equator
g1p = [] # 30 - 39 yrs Posterior
g2e = [] # 40 - 49 yrs Equator
g2p = [] # 40 - 49 yrs Posterior
g3e = [] # 50 - 59 yrs Equator
g3p = [] # 50 - 59 yrs Posterior
g4e = [] # 60 - 69 yrs Equator
g4p = [] # 60 - 69 yrs Posterior
g5e = [] # 70 - 79 yrs Equator
g5p = [] # 70 - 79 yrs Posterior
g6e = [] # 80 - 89 yrs Equator
g6p = [] # 80 - 89 yrs Posterior

# specific for paper 3
g8e = [] # 30 - 59 yrs Equator
g8p = [] # 30 - 59 yrs Posterior
g9e = [] # 60 - 89 yrs Equator
g9p = [] # 60 - 89 yrs Posterior

""" Plot Data """
for i in df.keys():
    if len(i) <= 2: # Only look at data traces ... [Row,Col]
        print(i)
        
        """ Data Traces """
        time['{}'.format(i)] = pd.to_numeric(df['{}'.format(i)].iloc[17:-1,0].reset_index(drop=True))
        extension['{}'.format(i)] =  pd.to_numeric(df['{}'.format(i)].iloc[17:-1,1].reset_index(drop=True))
        force['{}'.format(i)] =  pd.to_numeric(df['{}'.format(i)].iloc[17:-1,2].reset_index(drop=True))
        
        """ call data from pandas dataframe """
        HumanIDi = HumanID['{}'.format(i)]
        HumanAgei = HumanAge['{}'.format(i)]
        HumanGenderi = HumanGender['{}'.format(i)]
        HumanLeftRighti = HumanLeftRight['{}'.format(i)]
        HumanRegioni = HumanRegion['{}'.format(i)]
        PostMortemHrs_Mini = PostMortemHrs_Min['{}'.format(i)]
        DateOfDeathi = DateOfDeath['{}'.format(i)]
        TimeOfDeathi = TimeOfDeath['{}'.format(i)]
        EnucleationDatei = EnucleationDate['{}'.format(i)]
        EnucleationTimei = EnucleationTime['{}'.format(i)]
        DateOfTestingi = DateOfTesting['{}'.format(i)]
        TimeOfTestingi = TimeOfTesting['{}'.format(i)]
        DiameterPostAnti = DiameterPostAnt['{}'.format(i)]
        DiameterNasTempi = DiameterNasTemp['{}'.format(i)]
        SSii = SSi['{}'.format(i)]
        SSfi = SSf['{}'.format(i)]
        TMaxi = TFMax['{}'.format(i)]
        DispMaxi = DispMax['{}'.format(i)]
        FMaxi = FMax['{}'.format(i)]
        FSSi = FSS['{}'.format(i)]
        Slope10i = Slope10['{}'.format(i)]
        Rsqrd10i = Rsqrd10['{}'.format(i)]
        Slope20i = Slope20['{}'.format(i)]
        Rsqrd20i = Rsqrd20['{}'.format(i)]
        Slope30i = Slope30['{}'.format(i)]
        Rsqrd30i = Rsqrd30['{}'.format(i)]
        Slope0i = Slope0['{}'.format(i)]
        Rsqrd0i = Rsqrd0['{}'.format(i)]
        PeelVideoNamei = PeelVideoName['{}'.format(i)]
        PeelVideoHyperlinki = PeelVideoHyperlink['{}'.format(i)]
        VideoCommentsi = VideoComments['{}'.format(i)]
        LightMicroscopyImagesi = LightMicroscopyImages['{}'.format(i)]
        
        if HumanAgei >= 30 and HumanAgei < 40:
            if HumanRegioni == 'Equator':
                g1e.append(i)
                folder = '30_39_Equator'
            elif HumanRegioni == 'Posterior':
                g1p.append(i)
                folder = '30_39_Posterior'
        elif HumanAgei >= 40 and HumanAgei < 50:
            if HumanRegioni == 'Equator':
                g2e.append(i)
                folder = '40_49_Equator'
            elif HumanRegioni == 'Posterior':
                g2p.append(i)
                folder = '40_49_Posterior'
        elif HumanAgei >= 50 and HumanAgei < 60:
            if HumanRegioni == 'Equator':
                g3e.append(i)
                folder = '50_59_Equator'
            elif HumanRegioni == 'Posterior':
                g3p.append(i)
                folder = '50_59_Posterior'
        elif HumanAgei >= 60 and HumanAgei < 70:
            if HumanRegioni == 'Equator':
                g4e.append(i)
                folder = '60_69_Equator'
            elif HumanRegioni == 'Posterior':
                g4p.append(i)
                folder = '60_69_Posterior'
        elif HumanAgei >= 70 and HumanAgei < 80:
            if HumanRegioni == 'Equator':
                g5e.append(i)
                folder = '70_79_Equator'
            elif HumanRegioni == 'Posterior':
                g5p.append(i)
                folder = '70_79_Posterior'
        elif HumanAgei >= 80 and HumanAgei < 90:
            if HumanRegioni == 'Equator':
                g6e.append(i)
                folder = '80_89_Equator'
            elif HumanRegioni == 'Posterior':
                g6p.append(i)
                folder = '80_89_Posterior'
        
        # Category for the age group/region for paper 3
        if HumanAgei >= 30 and HumanAgei < 60:
            if HumanRegioni == 'Equator':
                g8e.append(i)
                folder2 = '30_59_Equator'
            elif HumanRegioni == 'Posterior':
                g8p.append(i)
                folder2 = '30_59_Posterior'
        elif HumanAgei >= 60 and HumanAgei < 90:
            if HumanRegioni == 'Equator':
                g9e.append(i)
                folder2 = '60_89_Equator'
            elif HumanRegioni == 'Posterior':
                g9p.append(i)
                folder2 = '60_89_Posterior'
        
        
        # Make the new folder to put figure
        folderPath = os.path.join(filePath, folder)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        
        # specific for paper 3
        folderPath2 = os.path.join(filePath, folder2)
        if not os.path.exists(folderPath2):
            os.makedirs(folderPath2)
        
        """ Plot force vs time and force vs disp """
        # Calculate gradient
        n = i
        gn = np.gradient(extension['{}'.format(n)],time['{}'.format(n)])
        
        tn = time['{}'.format(n)]
        fn = force['{}'.format(n)]
        dn = extension['{}'.format(n)]
        
        ssin = SSi['{}'.format(n)]
        ssfn = SSf['{}'.format(n)]
        
        # Check if steady state values exist, if they do create time/force array
        if str(ssin) == 'nan': 
            timeLocSSIn = np.nan
            timeLocSSFn = np.nan
            
            avgFnVal = np.nan
            
            sstnArray = np.array([ssin, ssfn])
            ssfnArray = np.array([avgFnVal, avgFnVal])
        else:
            # Location for steady state start/stop in the time array
            timeLocSSIn = tn.loc[lambda x: x == ssin].index.values[0]
            timeLocSSFn = tn.loc[lambda x: x == ssfn].index.values[0]
            
            # Average force value in between the steady state times
            avgFnVal = np.sum(fn[timeLocSSIn:timeLocSSFn])/len(range(timeLocSSIn,timeLocSSFn))
            
            sstnArray = np.array([ssin, ssfn]) # Steady-state time array
            ssfnArray = np.array([avgFnVal, avgFnVal]) # Steady-state force array
            ssdnArray = np.array([dn[timeLocSSIn], dn[timeLocSSFn]]) # Steady-state displacement array
        
        FMaxi = FMax['{}'.format(n)]
        if str(FMaxi) == 'nan':
            FmaxTimeLoc = np.nan
            tFmax = np.nan
            dFmax = np.nan # location in displacement where F = max
            
        else:
            FmaxTimeLoc = fn.loc[lambda x: x == FMaxi].index.values[0]
            tFmax = tn[FmaxTimeLoc]
            dFmax = dn[FmaxTimeLoc] # location in displacement where F = max
            
            # slope calculation for 10 seconds prior to the max peel force
            maxIndex = force['{}'.format(i)].loc[lambda x: x == FMax['{}'.format(i)]].index.values[0] # Location in the array for the max force
            x10 = extension['{}'.format(i)][maxIndex-100:maxIndex] # Array from maxIndex - 100 (10 sec) to location of max force
            y = force['{}'.format(i)][maxIndex-100:maxIndex] # Array from maxIndex - 100 (10 sec) to location of max force
            curveFit10, Params10, Rsqrd10val = Least_Squares(x10,y) # Perform least squares and return
            
            # slope calculation for 20 seconds prior to the max peel force
            x20 = extension['{}'.format(i)][maxIndex-200:maxIndex] # Array from maxIndex - 200 (20 sec) to location of max force
            y = force['{}'.format(i)][maxIndex-200:maxIndex] # Array from maxIndex - 200 (20 sec) to location of max force
            curveFit20, Params20, Rsqrd20val = Least_Squares(x20,y) # Perform least squares and return
            
            # slope calculation for 30 seconds prior to the max peel force
            x30 = extension['{}'.format(i)][maxIndex-300:maxIndex] # Array from maxIndex - 300 (30 sec) to location of max force
            y = force['{}'.format(i)][maxIndex-300:maxIndex] # Array from maxIndex - 300 (30 sec) to location of max force
            curveFit30, Params30, Rsqrd30val = Least_Squares(x30,y) # Perform least squares and return
            
            # Slope calculation from zero to max peel force
            x0 = extension['{}'.format(i)][0:maxIndex] # Array from 0 to location of max force
            y = force['{}'.format(i)][0:maxIndex] # Array from 0 to location of max force
            curveFit0, Params0, Rsqrd0val = Least_Squares(x0,y) # Perform least squares and return
        
        FSSi = avgFnVal
        
        """
        Plot force vs time
        """
        fign, axn = plt.subplots()
        
        # Plot data trace and Max & Steady-State vs time
        axn.plot(tn, fn*1e3,'-', color='k', linewidth=1, markersize=2, label = 'Trace #{}'.format(n))
        if str(FMaxi) == 'nan' and str(ssin) == 'nan':
            pass # used to be 'continue' but an error showed up
        
        if str(FMaxi) != 'nan':
            axn.plot(tFmax, FMaxi*1e3,'.', color='r', linewidth=1, markersize=20, label = 'Max Peel - {:.4f} (mN)'.format(FMaxi*1e3))
        
        if str(ssin) != 'nan':
            axn.plot(sstnArray, ssfnArray*1e3,'-', color='c', linewidth=3, markersize=2, label = 'Steady State - {:.4f} (mN)'.format(FSSi*1e3))
        
        axn.axhline(y=0, color='k')
        axn.set_xlabel('Time (sec)',fontsize=18)
        
        if str(FMaxi) == 'nan':
            axn.set_ylim(-0.5, fn.max()*1e3 + 1)
        else:
            axn.set_ylim(-0.5, FMaxi*1e3 + 1)
        axn.set_ylabel('Force (mN)',fontsize=18)
        axn.set_title(HumanIDi + ', ' + HumanGenderi + ', ' + 'Age: ' + str(HumanAgei) + ', ' + HumanLeftRighti + ', ' + HumanRegioni, fontsize=20)
        axn.grid(True, which='both')
        axn.legend(loc='best', prop={"size":12})
        fign.savefig(os.path.join(filePath,'Figures/' + 'Trace_{}'.format(str(n)) + '_F_vs_t.png'), dpi=300, bbox_inches='tight') # Save figure
        fign.savefig(os.path.join(filePath, folder, 'Trace_{}'.format(str(n)) + '_F_vs_t.png'), dpi=300, bbox_inches='tight') # Save figure
        fign.savefig(os.path.join(filePath, folder2, 'Trace_{}'.format(str(n)) + '_F_vs_t.png'), dpi=300, bbox_inches='tight') # Save figure
        plt.close()
        
        """
        Plot force vs displacement with slope
        """
        fign, axn = plt.subplots()
        
        # Plot data trace and Max & Steady-State vs displacement
        axn.plot(dn, fn*1e3,'-', color='k', linewidth=1, markersize=2, label = 'Trace #{}'.format(n), alpha = 0.3)
        
        if str(FMaxi) == 'nan' and str(ssin) == 'nan':
            pass # used to be 'continue' but an error showed up
        
        if str(FMaxi) != 'nan':
            axn.plot(dFmax, FMaxi*1e3,'.', color='r', linewidth=1, markersize=20, label = 'Max Peel - {:.4f} (mN)'.format(FMaxi*1e3))
            axn.plot(x0, curveFit0*1e3, '-', color='tab:blue', linewidth=1, label=r'Curve fit 0 (s) y = {:.4f}x + {:.4f} (mN), $r^2$ = {:.4f}'.format(Params0[1]*1e3, Params0[0]*1e3, Rsqrd0val))
            axn.plot(x30, curveFit30*1e3, '-', color='tab:orange', linewidth=2, label=r'Curve fit Max - 30 (s) y = {:.4f}x + {:.4f} (mN), $r^2$ = {:.4f}'.format(Params30[1]*1e3, Params30[0]*1e3, Rsqrd30val))
            axn.plot(x20, curveFit20*1e3, '-', color='tab:purple', linewidth=3, label=r'Curve fit Max - 20 (s) y = {:.4f}x + {:.4f} (mN), $r^2$ = {:.4f}'.format(Params20[1]*1e3, Params20[0]*1e3, Rsqrd20val))
            axn.plot(x10, curveFit10*1e3, '-', color='tab:green', linewidth=4, label=r'Curve fit Max - 10 (s) y = {:.4f}x + {:.4f} (mN), $r^2$ = {:.4f}'.format(Params10[1]*1e3, Params10[0]*1e3, Rsqrd10val))
        
        if str(ssin) != 'nan':
            axn.plot(ssdnArray, ssfnArray*1e3,'-', color='c', linewidth=3, markersize=2, label = 'Steady State - {:.4f} (mN)'.format(FSSi*1e3))
        
        axn.axhline(y=0, color='k')
        axn.set_xlabel('Disp (mm)',fontsize=18)
        
        if str(FMaxi) == 'nan':
            axn.set_ylim(-0.5, fn.max()*1e3 + 1)
        else:
            axn.set_ylim(-0.5, FMaxi*1e3 + 1)
        
        axn.set_ylabel('Force (mN)',fontsize=18)
        axn.set_title(HumanIDi + ', ' + HumanGenderi + ', ' + 'Age: ' + str(HumanAgei) + ', ' + HumanLeftRighti + ', ' + HumanRegioni, fontsize=20)
        axn.grid(True, which='both')
        axn.legend(loc='best', prop={"size":12})
        fign.savefig(os.path.join(filePath,'Figures/' + 'Trace_{}'.format(str(n)) + '_F_vs_disp.png'), dpi=300, bbox_inches='tight') # Save figure
        fign.savefig(os.path.join(filePath, folder, 'Trace_{}'.format(str(n)) + '_F_vs_disp.png'), dpi=300, bbox_inches='tight') # Save figure
        fign.savefig(os.path.join(filePath, folder2, 'Trace_{}'.format(str(n)) + '_F_vs_disp.png'), dpi=300, bbox_inches='tight') # Save figure
        plt.close()
        
        # Write the txt file with the force, extension, time data to the folder
        """ Print the Instron Data """
        print("\nWriting out the Instron data...")
        filename = os.path.join(filePath, folder, 'Trace_{}_Instron_Data'.format(str(n)) + '.txt')
        filename2 = os.path.join(filePath, folder2, 'Trace_{}_Instron_Data'.format(str(n)) + '.txt')
        outfile = open(filename,'w')
        outfile2 = open(filename2,'w')
        DataFile = ['Human ID:\t{}'.format(HumanIDi), 
                    'Human Age:\t{}'.format(HumanAgei), 
                    'Human Gender:\t{}'.format(HumanGenderi), 
                    'Human Left/Right:\t{}'.format(HumanLeftRighti), 
                    'Human Region:\t{}'.format(HumanRegioni), 
                    'Post Mortem Hrs_Min:\t{}'.format(PostMortemHrs_Mini), 
                    'Date of Death:\t{}'.format(DateOfDeathi), 
                    'Time of Death:\t{}'.format(TimeOfDeathi), 
                    'Enucleation Date:\t{}'.format(DateOfTestingi), 
                    'Enucleation Time:\t{}'.format(TimeOfTestingi), 
                    'Diameter Posterior Anterior (in):\t{}'.format(DiameterPostAnti), 
                    'Diameter Nasal Temporal (in):\t{}'.format(DiameterNasTempi), 
                    'SSi (s):\t{}'.format(SSii), 
                    'SSf (s):\t{}'.format(SSfi), 
                    'Time Max (s):\t{}'.format(TMaxi), 
                    'Disp Max (mm):\t{}'.format(DispMaxi), 
                    'FMax (mN):\t{}'.format(FMaxi*1e3), 
                    'FSS (mN):\t{}'.format(FSSi*1e3), 
                    'Slope 10 seconds before max to max force value (mN/m):\t{}'.format(Slope10i*1e3), 
                    'R^2 for linear regression 10 seconds before max:\t{}'.format(Rsqrd10i), 
                    'Slope 20 seconds before max to max force value (mN/m):\t{}'.format(Slope20i*1e3), 
                    'R^2 for linear regression 20 seconds before max:\t{}'.format(Rsqrd20i), 
                    'Slope 30 seconds before max to max force value (mN/m):\t{}'.format(Slope30i*1e3), 
                    'R^2 for linear regression 30 seconds before max:\t{}'.format(Rsqrd30i), 
                    'Slope from 0 to max force value (mN/m):\t{}'.format(Slope0i*1e3), 
                    'R^2 for linear regression from 0 to max:\t{}'.format(Rsqrd0i), 
                    'Peel Video Name:\t{}'.format(PeelVideoNamei), 
                    'Peel Video Hyperlink:\t{}'.format(PeelVideoHyperlinki), 
                    'Video Comments:\t{}'.format(VideoCommentsi), 
                    'Light Microscopy Images:\t{}'.format(LightMicroscopyImagesi), 
                    '\n', 
                    'Time (s)\tExtension (mm)\tForce (N)']
        
        HeaderWrite = '\n'.join(item for item in DataFile)
        outfile.write(HeaderWrite)
        outfile2.write(HeaderWrite)
        for i,j in enumerate(tn):
            line = '\n%f\t%f\t%f' % (j, dn[i], fn[i])
            outfile.write(line)
            outfile2.write(line)
        outfile.close()
        outfile2.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")

""" Plot the extension rate for last test """
fig2, ax2 = plt.subplots()
ax2.plot(tn, gn,'-', color='k', linewidth=1, markersize=2, label = '1')
ax2.set_xlim(0,2.5)
ax2.set_xlabel('Time (sec)',fontsize=18)
ax2.set_ylim(0,0.045)
ax2.set_ylabel('Velocity (mm/s)',fontsize=18)
ax2.set_title('Data Trace', fontsize=20)
ax2.grid(True, which='both')
lines = fig2.gca().get_lines()
show = [0]
legend1 = ax2.legend([lines[i] for i in show],[lines[i].get_label() for i in show], prop={"size":12}, loc='best', title = 'Trace')
ax2.add_artist(legend1)
plt.show()
plt.close()


# In[2]

""" Plot each age group data on top of one another """
os.chdir(filePath)
fileNames = next(os.walk('.'))[1]
print(fileNames)

for i in fileNames:
    if i == 'Figures' or i == 'StatisticsFigures':
        # skip these two folders
        continue
    
    elif i != '30_59_Equator' and i != '30_59_Posterior' and i != '60_89_Equator' and i != '60_89_Posterior':
        print(i, 'Age decade')
        subPath = os.path.join(filePath, i)
        (folderName, directory) = os.path.split(subPath)
        os.chdir(subPath)
        subTxtFiles = [x for x in glob.glob('*.txt')] # Posterior/Equator
        show = []
        maxVals = []
        
        color1 = iter(cm.Set1(np.linspace(0,1,len(subTxtFiles))))
        color2 = iter(cm.Set1(np.linspace(0,1,len(subTxtFiles))))
        color3 = iter(cm.Set1(np.linspace(0,1,len(subTxtFiles))))
        fign, axn = plt.subplots()
        for j,k in enumerate(subTxtFiles):
            c1 = next(color1)
            c2 = next(color2)
            c3 = next(color3)
            """ Read in the csv file """
            dfValsn = pd.read_csv(os.path.join(subPath, k), sep="\t", nrows=29, header=None, names=['Var','Attribute'])
            
            """ File Attributes """
            HID = dfValsn['Attribute'][0] # ID
            HAGE = dfValsn['Attribute'][1] # Age
            HG = dfValsn['Attribute'][2] # Gender
            HLR = dfValsn['Attribute'][3] # Left/Right
            HR = dfValsn['Attribute'][4] # Region
            HSSi = float(dfValsn['Attribute'][12]) # Steady state start time
            HSSf = float(dfValsn['Attribute'][13]) # Steady state final time
            HTMax = float(dfValsn['Attribute'][14]) # Time @ max force
            HFMax = float(dfValsn['Attribute'][16]) # Value at max force
            HFSS = float(dfValsn['Attribute'][17]) # Value at steady state
            
            dfn = pd.read_csv(os.path.join(subPath, k), sep="\t", header=30)
            dfn.columns = ['Time', 'Extension', 'Force']
            tn = dfn.Time
            dn = dfn.Extension
            force = dfn.Force
            
            # SS Array
            ssTimeArray = np.array([HSSi, HSSf])
            ssValArray = np.array([HFSS, HFSS])
            
            axn.plot(tn, force*1e3,'-', color=c1, linewidth=1, markersize=2, label = '{}, {}, Age: {}'.format(HID, HLR, HAGE), alpha = 1)
            if str(HFMax) == 'nan' and str(HSSi) == 'nan':
                continue
            
            if str(HFMax) != 'nan':
                axn.plot(HTMax, HFMax,'.', color=c2, linewidth=1, markersize=20, label = 'Max Peel - {:.4f} (mN)'.format(HFMax), path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
            
            if str(HSSi) != 'nan':
                axn.plot(ssTimeArray, ssValArray,'-', color=c3, linewidth=3, markersize=2, label = 'Steady State - {:.4f} (mN)'.format(HFSS), path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
            
            # append to the list for plot identification
            show.append(fign.gca().get_lines())
            
            # append the max value
            maxVals.append(HFMax)
            
        path, subFolder = os.path.split(subPath) # Extract the folder name
        axn.axhline(y=0, color='k')
        axn.set_xlabel('Time (sec)',fontsize=18)
        axn.set_ylim(-0.5, max(maxVals) + 1)
        axn.set_ylabel('Force (mN)',fontsize=18)
        axn.set_title(subFolder, fontsize=20)
        axn.grid(True, which='both')
        for i,j in enumerate(show):
            graphLine = []
            step = len(j)
            if i == 0:
                for k in range(0, len(j), 1):
                    graphLine.append(k)
            elif i > 0:
                for k in range(len(show[i-1]), len(j), 1):
                    graphLine.append(k)
            linesn = fign.gca().get_lines()
            legendn = axn.legend([linesn[i] for i in graphLine],[linesn[i].get_label() for i in graphLine], prop={"size":10}, loc=i+1, title = 'Data')
            axn.add_artist(legendn)
        plt.show()
        fign.savefig(os.path.join(subPath, '{}_All'.format(directory) + '_F_vs_t.png'), dpi=300, bbox_inches='tight') # Save figure
        plt.close()
        
    elif i == '30_59_Equator' or i == '30_59_Posterior' or i == '60_89_Equator' or i == '60_89_Posterior':
        # Plot the groups for paper 3 but don't include the max and steady state value (All legend items are in a single legend)
        print(i)
        subPath = os.path.join(filePath, i)
        (folderName, directory) = os.path.split(subPath)
        os.chdir(subPath)
        subTxtFiles = [x for x in glob.glob('*.txt')] # Posterior/Equator
        show = []
        maxVals = []
        
        color1 = iter(cm.rainbow(np.linspace(0,1,len(subTxtFiles))))
        color2 = iter(cm.rainbow(np.linspace(0,1,len(subTxtFiles))))
        color3 = iter(cm.rainbow(np.linspace(0,1,len(subTxtFiles))))
        fign, axn = plt.subplots()
        
        for j,k in enumerate(subTxtFiles):
            c1 = next(color1)
            c2 = next(color2)
            c3 = next(color3)
            """ Read in the csv file """
            dfValsn = pd.read_csv(os.path.join(subPath, k), sep="\t", nrows=29, header=None, names=['Var','Attribute'])
            
            """ File Attributes """
            HID = dfValsn['Attribute'][0] # ID
            HAGE = dfValsn['Attribute'][1] # Age
            HG = dfValsn['Attribute'][2] # Gender
            HLR = dfValsn['Attribute'][3] # Left/Right
            HR = dfValsn['Attribute'][4] # Region
            HSSi = float(dfValsn['Attribute'][12]) # Steady state start time
            HSSf = float(dfValsn['Attribute'][13]) # Steady state final time
            HTMax = float(dfValsn['Attribute'][14]) # Time @ max force
            HDispMax = float(dfValsn['Attribute'][15]) # Disp @ max force
            HFMax = float(dfValsn['Attribute'][16]) # Value at max force
            HFSS = float(dfValsn['Attribute'][17]) # Value at steady state
            
            dfn = pd.read_csv(os.path.join(subPath, k), sep="\t", header=30)
            dfn.columns = ['Time', 'Extension', 'Force']
            tn = dfn.Time
            dn = dfn.Extension # mm
            force = dfn.Force*1e3 # N ---> mN
            
            # SS Array
            ssTimeArray = np.array([HSSi, HSSf])
            ssValArray = np.array([HFSS, HFSS])
            ssDispArray = np.array([dn[tn == HSSi].values[0] if HSSi is not np.nan else np.nan, 
                                    dn[tn == HSSf].values[0] if HSSi is not np.nan else np.nan])
            
            axn.plot(dn, force,'-', color=c1, linewidth=1, markersize=2, label = '{}, {}, Age: {}'.format(HID, HLR, HAGE), alpha=0.3)
            
            if str(HFMax) != 'nan':
                maxIndex = force[tn == HTMax].index.values[0] # Location in the array for the max force
                
                # slope calculation for 10 seconds prior to the max peel force
                x10 = dn[maxIndex-100:maxIndex] # Array from maxIndex - 100 (10 sec) to location of max force
                y = force[maxIndex-100:maxIndex] # Array from maxIndex - 100 (10 sec) to location of max force
                curveFit10, Params10, Rsqrd10 = Least_Squares(x10,y) # Perform least squares and return
                
                # slope calculation for 20 seconds prior to the max peel force
                x20 = dn[maxIndex-200:maxIndex] # Array from maxIndex - 200 (20 sec) to location of max force
                y = force[maxIndex-200:maxIndex] # Array from maxIndex - 200 (20 sec) to location of max force
                curveFit20, Params20, Rsqrd20 = Least_Squares(x20,y) # Perform least squares and return
                
                # slope calculation for 30 seconds prior to the max peel force
                x30 = dn[maxIndex-300:maxIndex] # Array from maxIndex - 300 (30 sec) to location of max force
                y = force[maxIndex-300:maxIndex] # Array from maxIndex - 300 (30 sec) to location of max force
                curveFit30, Params30, Rsqrd30 = Least_Squares(x30,y) # Perform least squares and return
                
                # Slope calculation from zero to max peel force
                x0 = dn[0:maxIndex] # Array from 0 to location of max force
                y = force[0:maxIndex] # Array from 0 to location of max force
                curveFit0, Params0, Rsqrd0 = Least_Squares(x0,y) # Perform least squares and return
                
                # axn.plot(x0, curveFit0*1e3, ':', color='black', linewidth=1, label=r'_Curve fit 0 (s) y = {:.4f}x + {:.4f} (mN)'.format(Params0[1]*1e3, Params0[0]*1e3), alpha = 1)
                axn.plot(x30, curveFit30, '-', color='green', linewidth=2, label=r'_Curve fit Max - 30 (s) y = {:.4f}x + {:.4f} (mN), $r^2$ = {:.4f}'.format(Params30[1], Params30[0], Rsqrd30), alpha = 1)
                axn.plot(x20, curveFit20, '-', color='blue', linewidth=3, label=r'_Curve fit Max - 20 (s) y = {:.4f}x + {:.4f} (mN), $r^2$ = {:.4f}'.format(Params20[1], Params20[0], Rsqrd20), alpha = 1)
                axn.plot(x10, curveFit10, '-', color='red', linewidth=4, label=r'_Curve fit Max - 10 (s) y = {:.4f}x + {:.4f} (mN), $r^2$ = {:.4f}'.format(Params10[1], Params10[0], Rsqrd10), alpha = 1)
                
                # # Plot the max force value
                # axn.plot(HDispMax, HFMax,'.', color=c1, linewidth=1, markersize=20, label = 'Max Peel - {:.4f} (mN)'.format(HFMax), path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()]) # 
            
            # if str(HSSi) != 'nan':
            #     axn.plot(ssDispArray, ssValArray,'-', color=c1, linewidth=3, markersize=2, label = '_Steady State - {:.4f} (mN)'.format(HFSS), path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()], alpha = 0.5)
            
            # append to the list for plot identification
            show.append(fign.gca().get_lines())
            
            # append the max value
            maxVals.append(HFMax)
            
        path, subFolder = os.path.split(subPath) # Extract the folder name
        axn.axhline(y=0, color='k')
        axn.set_xlabel('Displacement (mm)',fontsize=18)
        axn.set_xlim(0, 9) # focus in on just ramp up
        axn.set_ylim(-0.5, max(maxVals) + 1)
        axn.set_ylabel('Force (mN)',fontsize=18)
        axn.set_title(subFolder, fontsize=20)
        # axn.grid(True, which='both')
        
        # where some data has already been plotted to ax
        handles, labels = axn.get_legend_handles_labels()
        
        # Manually add items to the legend
        fit_0 = mpatches.Patch(color='black', label=r'Curvefit (0 - $Time_{Max})$')
        fit_30 = mpatches.Patch(color='green', label=r'Curvefit (30 s before $Time_{Max})$')
        fit_20 = mpatches.Patch(color='blue', label=r'Curvefit (20 s before $Time_{Max})$')
        fit_10 = mpatches.Patch(color='red', label=r'Curvefit (10 s before $Time_{Max})$')
        
        # handles is a list, so append manual patch
        # handles.append(fit_0)
        handles.append(fit_30)
        handles.append(fit_20)
        handles.append(fit_10)
        
        axn.legend(handles=handles, loc='best')
        
        plt.show()
        fign.savefig(os.path.join(subPath, '{}_All'.format(directory) + '_F_vs_t.png'), dpi=300, bbox_inches='tight') # Save figure
        plt.close()
    
os.chdir(filePath)

