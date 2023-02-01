# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 22:28:58 2021

@author: Kiffer Creveling

This Python script does the following

    1) Select input parameters
    2) Generates the filename/description
    3) Calls Abaqus to create the .inp file w/ attributes & runs the job
    4) Creats a folder with the filename
    5) Extracts data from the Abaqus.odb file and creates two output files 
        (Field/Hist)
    6) Plots the data
    7) Moves all files that have the same filename 

"""

import os
import sys
import numpy as np
import pandas as pd
# import itertools as it # iteration tools (product fcn)
# from scipy import *
# import scipy.optimize as opt
import lmfit as lf
import pdb
import subprocess
import pprint

# Define the location of the Abaqus Working Directory
# specific folder path where this file is located
pythonScriptPath = os.getcwd()
abqWD, pythonFiles = os.path.split(pythonScriptPath) # split file path

# pythonScriptCreateINP_Run_ABQ (pS_ABQ)
pS_ABQ = os.path.join(pythonFiles, 'Cohesive_T3_EyeModel_Generate_Abaqus.py')
# pythonScriptExtract (pSE)
pSE = os.path.join(pythonFiles, 'Cohesive_T3_EyeModel_DataExtract.py')

# In[Job Info]

optE_V = True
optKsTsFE = False
sweep = False

if optE_V == True:
    optimization = 'E_V'
    
    """ Optimization of the vitreous using a tied interface """
    # If "True" then abaqus uses a tied interface between the nodes
    tieInterface = True
    
    """ Objective Function Flags """
    slopeFlag = True
    maxForceFlag = True
    ssForceFlag = False # Only used for damage
    
    """ Traction separation """
    DamageInitiation = False # If "False" then do not include damage initation
    DamageEvolution = False # If "False" then do not include damage evolution


if optKsTsFE == True:
    optimization = 'K_nnK_ssK_ttt_nt_st_tFE'
    
    """ Optimization of the vitreous using a tied interface """
    # If "True" then abaqus uses a tied interface between the nodes
    tieInterface = False
    
    """ Objective Function Flags """
    slopeFlag = False
    maxForceFlag = True
    ssForceFlag = True # Only used for damage
    
    """ Traction separation """
    DamageInitiation = True # If "False" then do not include damage initation
    DamageEvolution = True # If "False" then do not include damage evolution


if sweep == True:
    optimization = None
    
    """ Parametric sweep of the vitreous using a tied interface """
    # If "True" then abaqus uses a tied interface between the nodes
    tieInterface = False
    
    """ Objective Function Flags """
    slopeFlag = False
    maxForceFlag = True
    ssForceFlag = True # Only used for damage
    
    """ Traction separation """
    DamageInitiation = True # If "False" then do not include damage initation
    DamageEvolution = True # If "False" then do not include damage evolution


# # optimization info
# optList = []
# optList.append(None)
# optList.append('E_R')
# optList.append('E_V')
# optList.append('E_RE_V') # Retina and Vitreous Young's Modulus
# optList.append('K_nnK_ssK_tt') # Traction Separation Paramters
# Vitreous Young's Modulus and Traction Separation Parameters
# optList.append('E_VK_nnK_ssK_tt')
# optList.append('t_nt_st_t') # Damage initiation parameters
# optList.append('FE') # Damage evolution parameters
# optList.append('t_nt_st_tFE') # Damage initiation and evolution parameteres
# optList.append('K_nnK_ssK_ttt_nt_st_tFE') # All cohesive parameters
# All parameters except for retina young's modulus
# optList.append('E_VK_nnK_ssK_ttt_nt_st_tFE')

# Change to specific optimization parameter.  If 'None', no optimization
# optimization = optList[2]
# print('Optimization parameters = ', optimization)

# """ Optimization of the vitreous using a tied interface """
# # If "True" then abaqus uses a tied interface between the nodes
# tieInterface = False

# """ Objective Function Flags """
# slopeFlag = False
# maxForceFlag = True
# ssForceFlag = True # Only used for damage

# """ Traction separation """
# DamageInitiation = True # If "False" then do not include damage initation
# DamageEvolution = True # If "False" then do not include damage evolution

""" Objective Function Error Formulation """
objFunErr = []
objFunErr.append('Difference') # Experimental - Simulated
objFunErr.append('Ratio') # Experimental/Simulated
# (Experimental - Simulated)/Experimental
objFunErr.append('Relative uncertainty')
# Change to specific optimization parameter.  If 'None', no optimization
objErr = objFunErr[0]
print('Objective function error formulation = ', objErr)

# Calculation for error
ErrorCalculation = []
ErrorCalculation.append('two-point method') # Slope, Peak force, SS Force
ErrorCalculation.append('data-trace method') # interpolated array

errorMethod = ErrorCalculation[0]
print('Error method calculation = ', errorMethod)

''' Symmetry '''
# Split model in half and multiply output by 2
symmetry = True

''' Simplified '''
# Remove the rigid body on the plastic tab and glue
simplified = True

''' Gravity '''
# Turn gravity on/off
gravity = False # Keep off until model is updated

# In[Comparison Data Trace]
compareDataFolder = 'PeelDataCompare'
specificDataTrace = 'Trace_45_Instron_Data.txt' # Data trace number
timeBeforePeak = 40 # Default is 20 seconds
dataCompare = os.path.join(abqWD,compareDataFolder,specificDataTrace)
dfValsn = pd.read_csv(dataCompare, sep="\t", nrows=29, header=None, 
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
tn = dfn.Time
dn = dfn.Extension
df = dfn.Force # (N)

maxForceMeasured = HFMax # Value from data trace
maxSlopeMeasured = HSlope20 # slope from 20 seconds prior to max force value
SS_Measured = HFSS # simulated steady state force

# In[Functions]

if DamageInitiation == False and DamageEvolution == True:
    print('Unable to have DamageEvolution without DamageInitiation')
    sys.exit()

""" Tic Toc to determine runtime """
def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + 
              " seconds.")
        timeDiff = time.time() - startTime_for_tictoc
        return timeDiff
    else:
        print("Toc: start time not set")

try:
    os.environ.pop('PYTHONIOENCODING')
except KeyError:
    pass

# Import modules that plot/move all abq files to the new foldername
from ParameterSelection import ReadRAWDataTrace
from Cohesive_T3_Data_Plot import PlotAbqData
from Cohesive_T3_Residual import findResidual
from Cohesive_T3_CSMAXSCRT_MaxValue import CSMAXSCRTAbqData
from Move_ABQ_Files_To_Folder import MoveAbqFiles

newLine = '\n' + 77*'-' + '\n'

def jobAttributes():
    """
    Input: parameters used to create the filename and job description
    
    Output: namei, fileName, JobDescription
    """
    
    # Build the fileName
    fi = [] # initialize array
    fi.append(      '{}'.format(namei))
    fi.append(       'g') if gravity == True else ''
    fi.append(     'sym') if symmetry == True else ''
    fi.append(     't{}'.format(time))
    fi.append(    'E1{}'.format(e1Seedi[0]))
    fi.append(    'E2{}'.format(e2Seedi[0]))
    
    if simplified == False:
        fi.append(    'PT{}'.format(ptSeedi[0]))
        fi.append(     'G{}'.format(gSeedi[0]))
    
    fi.append(    'V1{}'.format(v1Seedi[0]))
    fi.append(    'V2{}'.format(v2Seedi[0]))
    fi.append(     'R{}'.format(rSeedi[0]))
    fi.append(     'F{}'.format(massScaleFactori[0]))
    fi.append(    'MS{}'.format(massScaleTimeIncrementi[0]))
    fi.append('RE{:.0e}'.format(RetinaYoungsModulus_i))
    
    if optimization is not None:
        if optimization.find('E_V') == -1:
            fi.append('VE{:.0e}'.format(VitreousYoungsModulus_i))
    
    # If optimization, get rid of the title (Not an integer anymore)
    if optimization is None:
        fi.append(    'Kn{}'.format(int(Knni[0])))
        fi.append(    'Ks{}'.format(int(Kssi[0])))
        fi.append(    'Kt{}'.format(int(Ktti[0])))
    
    # If True, then damage initation, If optimization, get rid of the title 
    # (Not an integer anymore)
    if DamageInitiation == True and optimization is None:
        fi.append(    'tn{}'.format(int(tni[0])))
        fi.append(    'ts{}'.format(int(tsi[0])))
        fi.append(    'tt{}'.format(int(tti[0])))
    
    # If True, then damage evolution, If optimization, get rid of the title 
    # (Not an integer anymore)
    if ((DamageInitiation == True) and (DamageEvolution == True) and 
        (optimization is None)):
        fi.append(    'FE{}'.format(int(FEi[0])))
    
    # .format(optimization))  optimization flag (I.e. RE, VE, Knn, Kss, 
    # tn, or none)
    fi.append(     'opt') if optimization is not None else ''
    fi.append(     'TIE') if tieInterface == True else ''
    
    if sweep == True:
        # get rid of all attributes because a sweep is taking place
        fi = fi[0]
        
    """ Build file name and description """
    fileName = ''.join(item for item in fi)
    # fix header so no decimals, math show up in title
    fileName = fileName.replace('+', '_').replace('-', '_').replace('.', '_')
    jobNameString = 'Job Name - {}'.format(fileName)
    
    # used for simplification of script
    # Large value
    multStrL = ('\n\tgeometric multiplier = 2**{}, \n\tbase value = {}, ' + 
                '\n\tmodel value = {}')
    # Small value
    multStrS = ('\n\tgeometric multiplier = 0.5**{}, \n\tbase value = {}, ' + 
                '\n\tmodel value = {}')
    
    # Build the model description
    si = [] # initialize array
    si.append(newLine)
    si.append('({}) = model name'.format(namei))
    si.append(jobNameString)
    si.append('(g) - Gravity') if gravity == True else si.append('No Gravity')
    # update name in list
    si.append('(sym) SYMMETRIC model (XY) Plane') if symmetry == True else ''
    # update name in list
    si.append('(t) Simulated time {} (s)'.format(time))
    
    # Eye Holder
    si.append(('(E1) Eye holder outside edge seed size (Max) (SINGLE BIAS):  ' 
               + multStrS + ' (m)').format(*e1Seedi))
    si.append(('(E2) Eye holder inside edge seed size (Min):  ' + multStrS + 
               ' (m)').format(*e2Seedi))
    
    # If simplified is in the title, get rid of glue and platic tab
    if simplified == False:
        si.append(('(PT) Plastic tab seed size:  ' + multStrS + 
                   ' (m)').format(*ptSeedi))
        si.append(('(G) Glue seed size:  ' + multStrS + ' (m)').format(*gSeedi))
    
    # Vitreous
    si.append(('(V1) Vitreous seed size max (side edge seed set)-' + 
               '(SINGLE BIAS):  ' + multStrS + ' (m)').format(*v1Seedi))
    si.append(('(V2) Vitreous seed size min (top edge in contact with ' + 
               'retina):  ' + multStrS + ' (m)').format(*v2Seedi))
    
    # Retina
    si.append(('(R) Retina seed size:  ' + multStrS + ' (m)').format(*rSeedi))
    
    # Mass scale factor
    si.append(('(F) Mass scale factor:  ' + multStrL + 
               '').format(*massScaleFactori))
   
    # Mass scale time increment
    si.append(('(MS) Mass scale time increment:  ' + multStrS + 
               ' (s)').format(*massScaleTimeIncrementi))
    
    # Material properties (Young's Modulus)
    si.append("(RE) Retina Young's Modulus:  model value = {} (Pa)"
              .format(RetinaYoungsModulus_i))
    si.append("(VE) Vitreous Young's Modulus:  model value = {} (Pa)"
              .format(VitreousYoungsModulus_i))
    
    # Cohesive traction parameters
    if tieInterface == False:
        si.append(('(Kn) Knn:  ' + multStrL + ' (Pa)').format(*Knni))
        si.append(('(Ks) Kss:  ' + multStrL + ' (Pa)').format(*Kssi))
        si.append(('(Kt) Ktt:  ' + multStrL + ' (Pa)').format(*Ktti))
    
    # If True, then damage initation
    if DamageInitiation == True:
        si.append(('(tn) tn:  ' + multStrL + ' (Pa)').format(*tni))
        si.append(('(ts) ts:  ' + multStrL + ' (Pa)').format(*tsi))
        si.append(('(tt) tt:  ' + multStrL + ' (Pa)').format(*tti))
    
    # If True, then damage evolution
    if DamageInitiation == True and DamageEvolution == True:
        si.append(('(FE) Fracture energy:  ' + multStrL + ' (J)').format(*FEi))
    
    # Optimization
    if optimization is not None:
        si.append('Optimization of {}'.format(optimization))
        si.append('Objective function error formulation is the ' + 
                  '{} calculation'.format(objErr))
        si.append('Objective error calculation is the {}'.format(errorMethod))
    
    if optimization == None:
        si.append('Parametric sweep')
        si.append('Objective function error formulation is the ' + 
                  '{} calculation'.format(objErr))
        si.append('Objective error calculation is the {}'.format(errorMethod))
    
    # Tied interface
    if tieInterface == True:
        si.append('Tied interface between the Retina and the Vitreous')
    
    # Data trace being compared for optimization
    si.append('The data trace being compared is:  {}'
              .format(specificDataTrace))
    
    # Time shift info as it is a new capability
    si.append('The time prior to the peak force time event used for ' + 
              'determining the linear region ' + 
              'was extended ({}) '.format(timeBeforePeak) +
              'seconds before the actual peak')
              
    si.append(newLine)
    
    # Job description
    jobDescription =  '\n'.join(item for item in si)
    
    print(newLine)
    print(fileName)
    print(newLine)
    print(jobDescription)
    
    # Write a .txt file with the file attributes
    outfile = open(os.path.join(abqWD, fileName +'.txt'),'w')
    line = ('The file name indicates what parameters were used to define ' + 
            'the model\n')
    outfile.write(line)
    line = '\n' + jobDescription + '\n'
    outfile.write(line)
    outfile.close()
    print(outfile)
    return namei, fileName, jobDescription


def GenerateAbaqusModels():
    """
    Function used to call Command Line (Windows Batch file)
    
    Parameters
    ----------
    fileName : abaqus job with paramters
    
    """
    # ----------------------- Step 2 -----------------------#
    # Generates the filename/description
    modelName, fileName, jobDescription = jobAttributes()
    
    # Strip job description from spaces and new lines
    # replace new lines, spaces, equal signs
    jobDescription = jobDescription.replace(' ', 'SPACE')
    jobDescription = jobDescription.replace('\n', 'NEWLINE')
    jobDescription = jobDescription.replace('\t', 'TAB')
    jobDescription = jobDescription.replace('=', 'EQUALSSIGN')
    
    print(newLine)
    
    # ----------------------- Step 3 -----------------------#
    # Calls Abaqus to create the job with the filename just created and 
    # run the job
    
    # Strip spaces and make strings
    ABQ = []
    ABQ.append(pS_ABQ) # python 2.7 script
    
    # gravity
    ABQ.append(','.join([i.strip(' ') for i in str(gravity).split(',')]))
    
    # symmetry
    ABQ.append(','.join([i.strip(' ') for i in str(symmetry).split(',')]))
    
    # Simplified model
    ABQ.append(','.join([i.strip(' ') for i in str(simplified).split(',')]))
    
    ABQ.append(modelName) # model name
    ABQ.append(fileName) # file name
    
    # time
    ABQ.append(','.join([i.strip(' ') for i in str(time).split(',')]))
    
    # eye holder seed size 1
    ABQ.append(','.join([i.strip(' ') for i in str(e1Seedi).split(',')]))
    
    # eye holder seed size 2
    ABQ.append(','.join([i.strip(' ') for i in str(e2Seedi).split(',')]))
    
    # plastic tab seed size
    ABQ.append(','.join([i.strip(' ') for i in str(ptSeedi).split(',')]))
    
    # glue seed size
    ABQ.append(','.join([i.strip(' ') for i in str(gSeedi).split(',')]))
    
    # vitreous seed 1 size
    ABQ.append(','.join([i.strip(' ') for i in str(v1Seedi).split(',')]))
    
    # vitreous seed 2 size
    ABQ.append(','.join([i.strip(' ') for i in str(v2Seedi).split(',')]))
    
    # retina seed size
    ABQ.append(','.join([i.strip(' ') for i in str(rSeedi).split(',')]))
    
    # mass scale factor
    ABQ.append(','.join([i.strip(' ') for i in 
                         str(massScaleFactori).split(',')]))
    
    # mass scale time
    ABQ.append(','.join([i.strip(' ') for i in 
                         str(massScaleTimeIncrementi).split(',')]))
    
    # Retina Young's Modulus
    ABQ.append(','.join([i.strip(' ') for i in 
                         str(RetinaYoungsModulus_i).split(',')]))
    
    # Vitreous Young's Modulus
    ABQ.append(','.join([i.strip(' ') for i in 
                         str(VitreousYoungsModulus_i).split(',')]))
    
    # Cohesive behavior
    ABQ.append(','.join([i.strip(' ') for i in str(Knni).split(',')])) # Knn
    ABQ.append(','.join([i.strip(' ') for i in str(Kssi).split(',')])) # Kss
    ABQ.append(','.join([i.strip(' ') for i in str(Ktti).split(',')])) # Ktt
    
    # DamageInitiation
    ABQ.append(','.join([i.strip(' ') for i in 
                         str(DamageInitiation).split(',')]))
    ABQ.append(','.join([i.strip(' ') for i in str(tni).split(',')])) # tn
    ABQ.append(','.join([i.strip(' ') for i in str(tsi).split(',')])) # ts
    ABQ.append(','.join([i.strip(' ') for i in str(tti).split(',')])) # tt
    
    # DamageEvolution
    ABQ.append(','.join([i.strip(' ') for i in 
                         str(DamageEvolution).split(',')]))
    ABQ.append(','.join([i.strip(' ') for i in str(FEi).split(',')])) # FE
    
    # Optimization None/optimized parameters
    ABQ.append(','.join([i.strip(' ') for i in str(optimization).split(',')]))
    
    # Tied interface
    ABQ.append(','.join([i.strip(' ') for i in str(tieInterface).split(',')]))
    
    # Model description
    ABQ.append(jobDescription)
    
    ABQ_parse_string = 'abaqus cae noGUI={} --' + (len(ABQ)-1)*' {}'
    
    # # Used for debugging, comment out to copy/paste output to cmd window
    # # to check and see if it works
    # print(ABQ_parse_string.format(*ABQ))
    # pdb.set_trace()
    
    cmd = subprocess.Popen(ABQ_parse_string.format(*ABQ), 
                            cwd=r'{}'.format(abqWD), stdin=subprocess.PIPE, 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                            shell=True).communicate()[0]
    
    print(newLine)
    print('Abaqus has generated the .inp and executed the job')
        
    # ----------------------- Step 4 -----------------------#
    # Creates a folder with the filename
    folderDirectory = os.path.join(abqWD, fileName)
    if not os.path.exists(folderDirectory):
        os.makedirs(folderDirectory)
    dataDirectory = os.path.join(folderDirectory, 'Output')
    if not os.path.exists(dataDirectory):
        os.makedirs(dataDirectory)
    figuresDirectory = os.path.join(dataDirectory, 'Figures')
    if not os.path.exists(figuresDirectory):
        os.makedirs(figuresDirectory)
    print(newLine)
    print('New file location:\n{} \n'.format(folderDirectory))
    
    # ----------------------- Step 5 -----------------------#
    """
    Extracts data from the Abaqus.odb file and creates two output files 
    (Field/Hist).  Create the name to be parsed into ABQ from the command 
    line through a subprocess
    """
    ABQ = []
    ABQ.append(pSE)
    ABQ.append(fileName)
    ABQ.append(gravity)
    ABQ.append(symmetry)
    ABQ.append(simplified)
    ABQ.append(DamageInitiation)
    ABQ.append(DamageEvolution)
    
    ABQ_parse_string = 'abaqus python' + len(ABQ)*' {}'
    
    # # # # Used for debugging, comment out to copy/paste output to cmd window
    # # # # to check and see if it works
    # print(ABQ_parse_string.format(*ABQ))
    # pdb.set_trace()
    
    cmd = subprocess.Popen(ABQ_parse_string.format(*ABQ), 
                            cwd=r'{}'.format(abqWD), stdin=subprocess.PIPE, 
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                            shell=True).communicate()[0]
    print(newLine)
    print('Abaqus has extracted Field/History output:  ' + 
          '\n{} \n'.format(dataDirectory))
    
    # ----------------------- Step 6 -----------------------#
    # Plot data and store it in the variable name folder under "Figures"
    print(fileName)
    print(dataDirectory)
    PlotAbqData(fileName, dataDirectory, dataCompare, DamageInitiation, 
                DamageEvolution)
    print(newLine)
    print('New data plots:\n{} \n'.format(figuresDirectory))
    
    # ----------------------- Step 7 -----------------------#
    # Move all abaqus files to the folder with the same name
    MoveAbqFiles(fileName, folderDirectory, abqWD)
    print(newLine)
    print('Files have been moved to:  \n{} \n'.format(dataDirectory))
    
    # ------------------- Step 8 (Error for minimization) -------------------#
    maxForceTime = 100 # s
    # slope is (mN/m)
    residVals = findResidual(fileName, dataDirectory, maxForceTime, 
                             dataCompare, objErr, slopeFlag, maxForceFlag, 
                             ssForceFlag, timeBeforePeak)
    # Unpack
    slopeSimulated =    residVals[0]
    maxForceSimulated = residVals[1]
    SSmeanSimulated =   residVals[2]
    SSmedianSimulated = residVals[3]
    y_new_exp_disp =    residVals[4]
    y_new_sim_disp =    residVals[5]
    
    # ------------Step 9 (Damage Initation for minimization) -------------#
    if DamageInitiation == True:
        retinaMaxUCRT, vitreousMaxUCRT = CSMAXSCRTAbqData(fileName, 
                                                          dataDirectory, 
                                                          maxForceTime, 
                                                          dataCompare)
    else:
        retinaMaxUCRT = np.nan
        vitreousMaxUCRT = np.nan
    
    # (return slope, force, and maxucrt @ specified displacement)
    fcnReturn = []
    fcnReturn.append(fileName)
    fcnReturn.append(slopeSimulated)
    fcnReturn.append(maxForceSimulated)
    fcnReturn.append(SSmeanSimulated)
    fcnReturn.append(SSmedianSimulated)
    fcnReturn.append(retinaMaxUCRT)
    fcnReturn.append(vitreousMaxUCRT)
    fcnReturn.append(y_new_exp_disp)
    fcnReturn.append(y_new_sim_disp)
    return fcnReturn


def writeOutputData(fileNameList):
    print("\nWriting out the Reaction Force data...")
    filename = os.path.join(abqWD, 'FEAAttributes' + '.txt')
    outfile = open(filename,'w')
    sep = '\t'
    Header = [] # List of items for the header
    Header.append('FileName')
    Header.append('Time')
    Header.append('E1')
    Header.append('E2')
    Header.append('PT')
    Header.append('G')
    Header.append('V1')
    Header.append('V2')
    Header.append('R')
    Header.append('F')
    Header.append('MS')
    Header.append('RE')
    Header.append('VE')
    Header.append('Knn')
    Header.append('Kss')
    Header.append('Ktt')
    Header.append('DamageInitiation')
    Header.append('tn')
    Header.append('ts')
    Header.append('tt')
    Header.append('DamageEvolution')
    Header.append('FE')
    Header.append('Optimization')
    Header.append('TIE')
    Header.append('errorListL2Norm')
    Header.append('ObjectiveFunction')
    Header.append('simTime')
    line = sep.join(item for item in Header)
    outfile.write(line)
    outfile.write('\n')
    outfile.write('\t'.join(str(item) for item in attributeArray_0))
    for i in list(fileNameList):
        outfile.write('\n')
        tempList = [str(i[0])] # filename
        for j in list(i[1]):
            tempList.append(str(j)) # file attributes
        tempList.append(str(i[2])) # sim time
        outfile.write('\t'.join(str(item) for item in tempList))
    outfile.close()
    print("\nDone!")
    print("\nThe output file will be named '{}".format(filename) + "'")
    print("\nIt will be in the same working directory as your Abaqus model\n")
    
    # Print File of tests ran in order
    print("\nWriting out the Reaction Force data...")
    filename = os.path.join(abqWD, 'FEAFileList' + '.txt')
    outfile = open(filename,'w')
    line = 'FileName'
    outfile.write(line)
    for i in list(fileNameList):
        line = '\n%s' % (i[0])
        outfile.write(line)
    outfile.close()
    print("\nDone!")
    print("\nThe output file will be named '{}".format(filename) + "'")
    print("\nIt will be in the same working directory as your Abaqus model\n")



if __name__ == '__main__':
    # Run the function
    
    # ----------------------- Step 1 -----------------------#
    # T3
    name = ['T3']
    
    paramSelect = ReadRAWDataTrace(dataCompare, abqWD, timeBeforePeak)
    
    t0, t1, tshift, fe = paramSelect # Unpack variables
    
    if t0 > tshift:
        # If the t1 value is greater than tshfit, use tshift for 
        # the simulation time
        # Shouldn't have to do this as this issue has been handeled
        t0 = tshift
        print('updated the time to be the shift value')
    
    # Determine which time to use (Max value or steady state)
    if optE_V == True:
        time = int(t0)
        FEValOpt = fe
        
    elif optKsTsFE == True or sweep == True:
        time = int(t1)
        FEValOpt = fe
    
    # Select input parameters
    # time = 97 # Simulation parameter time S25 shifted
    # time = 250 # Simulation parameter time
    
    ''' Optimized results using the updated optimization routine 2/11/21 
    using the larger vitreous strip model with first looking at the tied 
    interface between the vitreous and retina '''
    VitreousYoungsModulus_0 = 524.265652
    KnnValOpt = 26.312450336667535
    KssValOpt = 25.620054908304496
    KttValOpt = 27.028398378844223
    tnValOpt = 18.51999887865916
    tsValOpt = 17.98861859153288
    ttValOpt = 10.906247748496245
    # FEValOpt = -9.427062078905504
    
    """ Vitreous Young's Modulus """

    VitreousYoungsModulus_0 = 50.03617188307464 # optimized using Tie
    
    """ Retina Young's Modulus """
    RetinaYoungsModulus_0 = 11120.0 # Pa Optimized with the vitreous
    
    """ Eye holder inside edge """
    e1Seed_0 = 1 # Base seed
    e1SeedArray = [] # Array of multipliers
    n = 11 # number of increments
    for i in range(10, n):
        # Decrease mesh seed by a factor of 2
        e1SeedArray.append([i, e1Seed_0, e1Seed_0*(0.5)**i])
    
    """ Eye holder outside edge """
    # This will most likely never get smaller (saves computational time)
    e2Seed_0 = 1 # Base seed
    e2SeedArray = []
    n = 9 # number of increments
    for i in range(8, n):
        # Decrease mesh seed by a factor of 2
        e2SeedArray.append([i, e2Seed_0, e2Seed_0*(0.5)**i])
    
    """ Plastic tab """
    ptSeed_0 = 1 # Plastic tab seed size
    ptSeedArray = [] # Array of multipliers
    n = 7 # number of increments
    for i in range(6, n):
        # Decrease mesh seed by a factor of 2
        ptSeedArray.append([i, ptSeed_0, ptSeed_0*(0.5)**i])
    
    """ Glue """
    gSeed_0 = 1 # Glue seed size
    gSeedArray = [] # Array of multipliers
    n = 8 # number of increments
    for i in range(7, n):
        # Decrease mesh seed by a factor of 2
        gSeedArray.append([i, gSeed_0, gSeed_0*(0.5)**i])
    
    """ Vitreous """
    # smaller seed size
    v1Seed_0 = 1 # Vitreous (max seed size)
    v1SeedArray = [] # Array of multipliers
    # n = 30 # number of increments
    # for i in np.linspace(10, 12, n): # range(10, n):
    #     # Decrease mesh seed by a factor of 2
    #     v1SeedArray.append([i, v1Seed_0, v1Seed_0*(0.5)**i])
    
    # Comment out when parameters have been optimized
    v1ValOpt = 11.38 # (convergence value)
    v1SeedArray.append([v1ValOpt, v1Seed_0, v1Seed_0*(0.5)**v1ValOpt])
    
    # larger seed size (should be factor of 4 times smaller ## 2 numbers)
    v2Seed_0 = 1 # Vitreous (min seed size)
    v2SeedArray = [] # Array of multipliers
    # n = 9 # number of increments
    # for i in range(8, n):
    #     # Decrease mesh seed by a factor of 2
    #     v2SeedArray.append([i, v2Seed_0, v2Seed_0*(0.5)**i])
    
    # Comment out when parameters have been optimized
    v2ValOpt = 8
    v2SeedArray.append([v2ValOpt, v2Seed_0, v2Seed_0*(0.5)**v2ValOpt])
    
    """ Retina """
    rSeed_0 = 1 # Base seed
    rSeedArray = [] # Array of multipliers
    # n = 30 # number of increments
    # for i in np.linspace(10, 13.5, n): # range(10, n):
    #     # Decrease mesh seed by a factor of 2
    #     rSeedArray.append([i, rSeed_0, rSeed_0*(0.5)**i])
        
    rValOpt = 11.3275 # (convergence value)
    rSeedArray.append([rValOpt, rSeed_0, rSeed_0*(0.5)**rValOpt])
    
    """ mass scale factor """
    massScaleFactor_0 = 1
    massScaleFactorArray = [] # Array of multipliers
    n = 1 # number of increments
    for i in range(0, n):
        # Increase by a factor of 2
        massScaleFactorArray.append([i, massScaleFactor_0, 
                                     massScaleFactor_0*2**i])
    
    """ mass scale time increment """
    massScaleTimeIncrement_0 = 1
    massScaleTimeArray = [] # multiplier and value
    n = 8 # number of increments
    for i in range(7, n):
        # Decrease by a factor of 2
        massScaleTimeArray.append([i, massScaleTimeIncrement_0, 
                                   massScaleTimeIncrement_0*(0.5)**i])
    
    if massScaleTimeIncrement_0 == 0:
        print('No Mass Scaling... This will take a while...ABAQUS is ' + 
              'deciding for us')
    
    """ Knn """
    Knn_0 = 1
    KnnArray = [] # Array of multipliers
    # n = 31 # number of increments # 23 works when R = 2e3, and V = 736 Pa
    # for i in range(30, n):
    #     # Increase by a factor of 2
    #     KnnArray.append([i, Knn_0, Knn_0*(2)**i])
    
    # Comment out when parameters have been optimized
    KnnArray.append([KnnValOpt, Knn_0, Knn_0*(2)**KnnValOpt])
    
    """ Kss """
    Kss_0 = 1
    KssArray = [] # Array of multipliers
    # n = 31
    # for i in range(30, n):
    #     # Increase by a factor of 2
    #     KssArray.append([i, Kss_0, Kss_0*(2)**i])
    
    # Comment out when parameters have been optimized
    KssArray.append([KssValOpt, Kss_0, Kss_0*(2)**KssValOpt])
    
    """ Ktt """
    Ktt_0 = 1
    KttArray = [] # Array of multipliers
    # n = 31
    # for i in range(30, n):
    #     # Increase by a factor of 2
    #     KttArray.append([i, Ktt_0, Ktt_0*(2)**i])
    
    # Comment out when parameters have been optimized
    KttArray.append([KttValOpt, Ktt_0, Ktt_0*(2)**KttValOpt])
    
    """ tn """
    tn_0 = 1
    tnArray = [] # Array of multipliers
    # n = 10 # 10 works when using max stress criteria
    # for i in range(9, n):
    #     # Increase by a factor of 2
    #     tnArray.append([i, tn_0, tn_0*(2)**i])
    
    # Comment out when parameters have been optimized
    tnArray.append([tnValOpt, tn_0, tn_0*(2)**tnValOpt])
    
    """ ts """
    ts_0 = 1
    tsArray = [] # Array of multipliers
    # n = 10
    # for i in range(9, n):
    #     # Increase by a factor of 2
    #     tsArray.append([i, ts_0, ts_0*(2)**i])
    
    # Comment out when parameters have been optimized
    tsArray.append([tsValOpt, tn_0, tn_0*(2)**tsValOpt])
        
    """ tt """
    tt_0 = 1
    ttArray = [] # Array of multipliers
    # n = 10
    # for i in range(9, n):
    #     # Increase by a factor of 2
    #     ttArray.append([i, tt_0, tt_0*(2)**i])
    
    # Comment out when parameters have been optimized
    ttArray.append([ttValOpt, tn_0, tn_0*(2)**ttValOpt])
    
    """ FE """
    FE_0 = 1
    FEArray = [] # Array of multipliers
    # n = -8
    # for i in range(-9, n):
    #     # Increase by a factor of 2
    #     FEArray.append([i, FE_0, FE_0*(2)**i])
    
    FEArray.append([FEValOpt, FE_0, FE_0*(2)**FEValOpt])
    
    errorList = np.nan # initial error
    slopeList = np.nan # Initial slope
    FmaxList = np.nan # Initial max peel force
    FSSList = np.nan # Initial steady-state peel force
    
    """ Attribute Array Initial Values """
    attributeArray_0 = []
    attributeArray_0.append('BaseVals')
    attributeArray_0.append(time)
    attributeArray_0.append(e1Seed_0)
    attributeArray_0.append(e2Seed_0)
    attributeArray_0.append(ptSeed_0)
    attributeArray_0.append(gSeed_0)
    attributeArray_0.append(v1Seed_0)
    attributeArray_0.append(v2Seed_0)
    attributeArray_0.append(rSeed_0)
    attributeArray_0.append(massScaleFactor_0)
    attributeArray_0.append(massScaleTimeIncrement_0)
    attributeArray_0.append(RetinaYoungsModulus_0)
    attributeArray_0.append(VitreousYoungsModulus_0)
    attributeArray_0.append(Knn_0)
    attributeArray_0.append(Kss_0)
    attributeArray_0.append(Ktt_0)
    attributeArray_0.append(DamageInitiation)
    attributeArray_0.append(tn_0)
    attributeArray_0.append(ts_0)
    attributeArray_0.append(tt_0)
    attributeArray_0.append(DamageEvolution)
    attributeArray_0.append(FE_0)
    attributeArray_0.append(optimization)
    attributeArray_0.append(tieInterface)
    attributeArray_0.append(errorList)
    attributeArray_0.append(objErr)
    attributeArray_0.append(slopeList)
    attributeArray_0.append(FmaxList)
    attributeArray_0.append(FSSList)
    attributeArray_0.append('simTime')
    
    
    fileNameList = [] # List of files
    counter = 0
    
    if optimization is not None:
        """ If the optimization variable is not "None" then optimize the 
        specific variable beins passed through """
        
        name = name[0]
        
        # BondStatus = True # interested in bonding
        
        # # post damage failure model (If False, ignore Ktt, ts, and tn, 
        # # otherwise include them)
        # pdfm = False
        
        # Optimization method
        # optName = 'NM' # Nelder-mead
        # optName = 'P' # Powell
        optName = 'C' # COBYLA
        # optName = 'L' # LBFGSB
        # optName = 'T' # Truncated Newton
        # optName = 'S' # SLSQP
        # optName - 'TC' # Trust-Constr
        
        name0 = '_'.join([name, optName]) # used for optimization
        
        def FEA_Residual(pars, data=None):
            # Global variables
            global counter
            global name
            global name0
            global fileNameList
            global time
            global namei
            global e1Seedi
            global e2Seedi
            global ptSeedi
            global gSeedi
            global v1Seedi
            global v2Seedi
            global rSeedi
            global massScaleFactori
            global massScaleTimeIncrementi
            global RetinaYoungsModulus_i
            global VitreousYoungsModulus_i
            global Knni
            global Kssi
            global Ktti
            global tni
            global tsi
            global tti
            global FEi
            
            # Parameters used for optimization
            global errorList
            
            print('Iteration # ', counter)
            
            tic() # Start time
            
            e1Seedi = e1SeedArray[0] # Default array
            e2Seedi = e2SeedArray[0] # Default array
            ptSeedi = ptSeedArray[0] # Default array
            gSeedi = gSeedArray[0] # Default array
            v1Seedi = v1SeedArray[0] # Default array
            v2Seedi = v2SeedArray[0] # Default array
            rSeedi = rSeedArray[0] # Default array
            massScaleFactori = massScaleFactorArray[0] # Default array
            massScaleTimeIncrementi = massScaleTimeArray[0] # Default array
            RetinaYoungsModulus_i = RetinaYoungsModulus_0 # Default value
            VitreousYoungsModulus_i = VitreousYoungsModulus_0 # Default value
            Knni = KnnArray[0] # Default array
            Kssi = KssArray[0] # Default array
            Ktti = KttArray[0] # Default array
            tni = tnArray[0] # Default array
            tsi = tsArray[0] # Default array
            tti = ttArray[0] # Default array
            FEi = FEArray[0] # Default array
            
            # Extract the unknown parameters from the pars class variable
            # Determine the multiplier for the title
            for key, value in pars.items():
                
                if key.find('ER') >= 0:
                    """ Retina Young's Modulus """
                    val = value.value
                    RetinaYoungsModulus_i = val
                
                elif key.find('EV') >= 0:
                    """ Vitreous Young's Modulus """
                    val = value.value
                    VitreousYoungsModulus_i = val
                
                elif key.find('Knn') >= 0:
                    """ Knn """
                    val = value.value
                    mult = np.log(val)/np.log(2) # multiplier
                    Knni = [mult, Knn_0, val]
                
                elif key.find('Kss') >= 0:
                    """ Kss """
                    val = value.value
                    mult = np.log(val)/np.log(2) # multiplier
                    Kssi = [mult, Kss_0, val]
                
                elif key.find('Ktt') >= 0:
                    """ Ktt """
                    val = value.value
                    mult = np.log(val)/np.log(2) # multiplier
                    Ktti = [mult, Ktt_0, val]
                
                elif key.find('tn') >= 0:
                    """ tn """
                    val = value.value
                    mult = np.log(val)/np.log(2) # multiplier
                    tni = [mult, tn_0, val]
                
                elif key.find('ts') >= 0:
                    """ ts """
                    val = value.value
                    mult = np.log(val)/np.log(2) # multiplier
                    tsi = [mult, ts_0, val]
                
                elif key.find('tt') >= 0:
                    """ tt """
                    val = value.value
                    mult = np.log(val)/np.log(2) # multiplier
                    tti = [mult, tt_0, val]
                
                elif key.find('FE') >= 0:
                    """ FE """
                    val = value.value
                    mult = np.log(val)/np.log(2) # multiplier
                    FEi = [mult, FE_0, val]
            
            # Keep track of simulation results by unique names with the count
            # number.  Comment out the second part to save file space if you
            # are not interested in saving every single simulation
            namei = name0 #+ str(counter)
            
            # Error of the simulation
            L2Normi = np.sqrt(np.dot(errorList, errorList))
            
            # multipliers to be appended to the output file to show changes 
            # in parameters
            aAM = [] # attributeArrayMultipliar
            aAM.append(time)
            aAM.append(e1Seedi[0])
            aAM.append(e2Seedi[0])
            aAM.append(ptSeedi[0])
            aAM.append(gSeedi[0])
            aAM.append(v1Seedi[0])
            aAM.append(v2Seedi[0])
            aAM.append(rSeedi[0])
            aAM.append(massScaleFactori[0])
            aAM.append(massScaleTimeIncrementi[0])
            aAM.append(RetinaYoungsModulus_i)
            aAM.append(VitreousYoungsModulus_i)
            aAM.append(Knni[0])
            aAM.append(Kssi[0])
            aAM.append(Ktti[0])
            aAM.append(DamageInitiation)
            aAM.append(tni[0])
            aAM.append(tsi[0])
            aAM.append(tti[0])
            aAM.append(DamageEvolution)
            aAM.append(FEi[0])
            aAM.append(optimization)
            aAM.append(tieInterface)
            aAM.append(L2Normi)
            aAM.append(objErr)
            
            # Call the function
            # Runs jobs and saves file names
            funReturn = GenerateAbaqusModels()
            fileName =           funReturn[0]
            maxSlopeSimulated =  funReturn[1]
            maxForceSimulated =  funReturn[2]
            SSmeanSimulated =    funReturn[3]
            SSmedianSimulated =  funReturn[4]
            retinaMaxUCRT =      funReturn[5]
            vitreousMaxUCRT =    funReturn[6]
            y_new_exp_disp =     funReturn[7]
            y_new_sim_disp =     funReturn[8]
            
            # add the simulated outputs to the data file
            aAM.append(maxSlopeSimulated)
            aAM.append(maxForceSimulated)
            aAM.append(SSmedianSimulated)
            
            # Determine the measure of error used for optimization
            # Let the data trace being passed in act as the comparison
            maxSlopeMeasured, maxForceMeasured = data
            
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
            L2Normi = np.sqrt(np.dot(errorList, errorList))
            
            # Calculate residual
            residual = y_new_exp_disp - y_new_sim_disp # residual
            
            # Calculate L2Norm
            L2Norm = np.sqrt(np.dot(residual, residual))
            
            simulationTime = toc() # Determine run time
            # apends the fileName & File Attributes
            fileNameList.append([fileName, aAM, 
                                 simulationTime])
            print('{} Error calculation: '.format(objErr), errorList)
            print('L2 norm objective calculation', L2Normi)
            print('L2 Norm residual', L2Norm)
            print('Done\n\n\n')
            counter += 1
            
            # Determine which calculation is going to be used for optimization
            if errorMethod == 'two-point method':
                FEA_Residual = errorList
            elif errorMethod == 'data-trace method':
                FEA_Residual = residual
                     
            return FEA_Residual
        
        maxFuncEval = 200
        tolVal = 1e-4
        
        # Use the data variable to input the max slope and force from the 
        # known data trace
        data = [maxSlopeMeasured, maxForceMeasured]
        
        # Initial, Upper, and Lower bounds for parameters
        
        # Young's Modulus - Retina
        ER_i = 5000 # Pa
        ER_LB = 50 # Pa
        ER_UB = 11000 # Pa
        
        # Young's Modulus - Vitreous
        EV_i = 172 # Pa (Prony series calculation)
        # EV_i = 500 # Pa (Trying higher initial guess)
        EV_LB = 50 # Pa
        EV_UB = 2100 # Pa
        # EV_UB = 400 # Pa (Lowering the upper bound)
        
        # Traction-Separation Behavior
        # Knn_i = 2**18 # Stress [Pa]
        # Knn_i = 2**26.32642676301851 # better optimized guess
        Knn_i = 2**20.872765304828103 # Stress [Pa] # different guess Low E_v
        # Knn_LB = 2**22 # Stress [Pa]
        Knn_LB = 2**10 # Stress [Pa] # Try lowering the bound
        Knn_UB = 2**28 # Stress [Pa]
        
        # Kss_i = 2**18 # Stress [Pa]
        # Kss_i = 2**27.387981486684094 # better optimized guess
        Kss_i = 2**26.094732037712763 # different guess Low E_v
        # Kss_LB = 2**22 # Stress [Pa]
        Kss_LB = 2**10 # Stress [Pa] # Try lowering the bound
        Kss_UB = 2**28 # Stress [Pa]
        
        # Ktt_i = 2**18 # Stress [Pa]
        # Ktt_i = 2**27.88464867824286 # better optimized guess
        Ktt_i = 2**26.20110650892766 # different guess Low E_v
        # Ktt_LB = 2**22 # Stress [Pa]
        Ktt_LB = 2**10 # Stress [Pa] # Try lowering the bound
        Ktt_UB = 2**28# # Stress [Pa]
        
        # Damage Initiation Behavior
        # tn_i = 2**9 # Stress [Pa]
        # tn_i = 2**18.51999887865916 # better optimized guess
        tn_i = 2**9.712181223168551 # different guess Low E_v
        tn_LB = 2**3 # Stress [Pa]
        tn_UB = 2**20 # Stress [Pa] 11 before
        
        # ts_i = 2**9 # Stress [Pa]
        # ts_i = 2**17.98861859153288 # better optimized guess
        ts_i = 2**9.931687876075074 # different guess Low E_v
        ts_LB = 2**3 # Stress [Pa]
        ts_UB = 2**20 # Stress [Pa] 11 before
        
        # tt_i = 2**9 # Stress [Pa]
        # tt_i = 2**10.906247748496245 # better optimized guess
        tt_i = 2**9.022372079206395 # different guess Low E_v
        tt_LB = 2**3 # Stress [Pa]
        tt_UB = 2**15 # Stress [Pa] 11 before
        
        # Damage Evolution Behavior
        # FE_i = 3.738925970000001e-6 # Energy [J]  ~ -18.028944662923816 # S25
        FE_i = 1.929e-6 # Energy [J]  S47
        # FE_i = 2**-9.427062078905504 # better optimized guess
        FE_LB = 2**-30 # 2**0 # Energy [J]
        # FE_UB = 2**-8 # Energy [J] # small bounds on energy
        FE_UB = 2**0 # Energy [J] # increase bounds
        
        # Specify parameters
        fit_params = lf.Parameters() # intialize the class for parameters
        
        # Retina young's modulus
        if optimization.find('E_R') >= 0:
            fit_params.add('ER', value = ER_i, min=ER_LB, max=ER_UB, vary=True)
        
        # Vitreous Young's Modulus
        if optimization.find('E_V') >= 0:
            fit_params.add('EV', value = EV_i, min=EV_LB, max=EV_UB, vary=True)
        
        # parameter for making the retina stiffer than the vitreous
        if optimization.find('E_R') >= 0 and optimization.find('E_V') >= 0:
            fit_params.add('StiffDelta', value = 0.01, min=0, vary=True)
            # Constraint to allow vitreous to be not as stiff as the retina
            fit_params.add('stiffnessConstraint', expr = 'EV - StiffDelta')
        
        # Knn
        if optimization.find('K_nn') >= 0:
            fit_params.add('Knn', value = Knn_i, min=Knn_LB, max=Knn_UB, 
                           vary=True)
        
        # Kss
        if optimization.find('K_ss') >= 0:
            fit_params.add('Kss', value = Kss_i, min=Kss_LB, max=Kss_UB, 
                           vary=True)
        
        # Ktt
        if optimization.find('K_tt') >= 0:
            fit_params.add('Ktt', value = Ktt_i, min=Ktt_LB, max=Ktt_UB, 
                           vary=True)
        
        # tn
        if optimization.find('t_n') >= 0:
            fit_params.add('tn', value = tn_i, min=tn_LB, max=tn_UB, 
                           vary=True)
        
        # ts
        if optimization.find('t_s') >= 0:
            fit_params.add('ts', value = ts_i, min=ts_LB, max=ts_UB, 
                           vary=True)
        
        # tt
        if optimization.find('t_t') >= 0:
            fit_params.add('tt', value = tt_i, min=tt_LB, max=tt_UB, 
                           vary=True)
        
        # FE
        if optimization.find('FE') >= 0:
            fit_params.add('FE',  value = FE_i, min=FE_LB, max=FE_UB, 
                           vary=True)
        
        # Set up minimization class
        minClass = lf.Minimizer(FEA_Residual, fit_params, 
                                fcn_kws={'data': data}, 
                                max_nfev = maxFuncEval) # fcn_args=(x,), 
        
        # (Different methods can be used here) Uses an array
        # out = minClass.leastsq() # Levenberg-Marquardt
        
        # single scalar value
        # out = minClass.scalar_minimize(method='Nelder-Mead', tol=tolVal)
        
        # single scalar value (if the objective function returns an array, 
        # the sum of the squares of the array will be used (L2Norm))
        out = minClass.scalar_minimize(method='Cobyla', tol=tolVal) 
        
        lf.report_fit(out) # modelpars=p_true,  show_correl=True
        
        # Write data to txt files
        writeOutputData(fileNameList)
    
    else:
    
        # Number of simulations to perform (Simulation Batch Total)
        SBT = []
        SBT.append(len(name))
        SBT.append(len(e1SeedArray))
        SBT.append(len(e2SeedArray))
        SBT.append(len(ptSeedArray))
        SBT.append(len(gSeedArray))
        SBT.append(len(v1SeedArray))
        SBT.append(len(v2SeedArray))
        SBT.append(len(rSeedArray))
        SBT.append(len(massScaleFactorArray))
        SBT.append(len(massScaleTimeArray))
        SBT.append(len(KnnArray))
        SBT.append(len(KssArray))
        SBT.append(len(KttArray))
        SBT.append(len(tnArray))
        SBT.append(len(tsArray))
        SBT.append(len(ttArray))
        SBT.append(len(FEArray))
        
        ZipArray = []
        ZipArray.append(max(SBT)*name)
        ZipArray.append(max(SBT)*e1SeedArray)
        ZipArray.append(max(SBT)*e2SeedArray)
        ZipArray.append(max(SBT)*ptSeedArray)
        ZipArray.append(max(SBT)*gSeedArray)
        ZipArray.append(max(SBT)*v1SeedArray)
        ZipArray.append(max(SBT)*v2SeedArray)
        ZipArray.append(max(SBT)*rSeedArray)
        ZipArray.append(max(SBT)*massScaleFactorArray)
        ZipArray.append(max(SBT)*massScaleTimeArray)
        ZipArray.append(max(SBT)*KnnArray)
        ZipArray.append(max(SBT)*KssArray)
        ZipArray.append(max(SBT)*KttArray)
        ZipArray.append(max(SBT)*tnArray)
        ZipArray.append(max(SBT)*tsArray)
        ZipArray.append(max(SBT)*ttArray)
        ZipArray.append(max(SBT)*FEArray)
        
        # Iterate over the different combinations of parameters
        # If varying one parameter, then use iter.product(items in list...)
        # If varying multiple parameters, use zip*max(SBT)*items in list...)
        
        for (namei, 
             e1Seedi, 
             e2Seedi, 
             ptSeedi, 
             gSeedi, 
             v1Seedi, 
             v2Seedi, 
             rSeedi, 
             massScaleFactori, 
             massScaleTimeIncrementi, 
             Knni, 
             Kssi, 
             Ktti, 
             tni, 
             tsi, 
             tti, 
             FEi) in zip(*ZipArray):
            tic() # Start time
            counter += 1
            print(counter, 'of ', max(*SBT))
            
            namei = namei + '_{}'.format(counter)
            
            # set the i'th value to the initial value (Updated in 
            # optimization algorithm)
            RetinaYoungsModulus_i = RetinaYoungsModulus_0
            VitreousYoungsModulus_i = VitreousYoungsModulus_0
            
            # Call the function
            # Runs jobs and saves file names
            funReturn = GenerateAbaqusModels()
            fileName =           funReturn[0]
            maxSlopeSimulated =  funReturn[1]
            maxForceSimulated =  funReturn[2]
            SSmeanSimulated =    funReturn[3]
            SSmedianSimulated =  funReturn[4]
            retinaMaxUCRT =      funReturn[5]
            vitreousMaxUCRT =    funReturn[6]
            y_new_exp_disp =     funReturn[7]
            y_new_sim_disp =     funReturn[8]
            
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
            
            # Error of the simulation
            L2Normi = np.sqrt(np.dot(errorList, errorList))
            
            # Calculate residual
            residual = y_new_exp_disp - y_new_sim_disp # residual
            
            # Calculate L2Norm
            L2Norm = np.sqrt(np.dot(residual, residual))
            
            # multipliers to be appended to the output file to show changes 
            # in parameters
            aAM = [] # attributeArrayMultipliar
            aAM.append(time)
            aAM.append(e1Seedi[0])
            aAM.append(e2Seedi[0])
            aAM.append(ptSeedi[0])
            aAM.append(gSeedi[0])
            aAM.append(v1Seedi[0])
            aAM.append(v2Seedi[0])
            aAM.append(rSeedi[0])
            aAM.append(massScaleFactori[0])
            aAM.append(massScaleTimeIncrementi[0])
            aAM.append(RetinaYoungsModulus_i)
            aAM.append(VitreousYoungsModulus_i)
            aAM.append(Knni[0])
            aAM.append(Kssi[0])
            aAM.append(Ktti[0])
            aAM.append(DamageInitiation)
            aAM.append(tni[0])
            aAM.append(tsi[0])
            aAM.append(tti[0])
            aAM.append(DamageEvolution)
            aAM.append(FEi[0])
            aAM.append(optimization)
            aAM.append(tieInterface)
            aAM.append(L2Normi)
            aAM.append(objErr)
            aAM.append(maxSlopeSimulated)
            aAM.append(maxForceSimulated)
            aAM.append(SSmedianSimulated)
            
            simulationTime = toc() # Determine run time
            # apends the fileName & File Attributes
            fileNameList.append([fileName, aAM, 
                                 simulationTime])
            print('{} Error calculation: '.format(objErr), errorList)
            print('L2 norm objective calculation', L2Normi)
            print('L2 Norm residual', L2Norm)
            print('Done')
            
        # Write data to txt files
        writeOutputData(fileNameList)

