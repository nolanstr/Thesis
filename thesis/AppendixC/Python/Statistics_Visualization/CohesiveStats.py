# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 17:25:44 2021

@author: Kiffer
"""

import pandas as pd
import numpy as np
import seaborn as sns
from statannot import add_stat_annotation
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pdb
import os
import glob
import re

cwd = os.getcwd()

# In[Peel test data]]

fp = os.path.join(cwd, 'Results', 'ExpPeelDataCompare')

# Grab all files that have "trace_"
fileList = glob.glob(os.path.join(fp, "trace_*"))

finalAttr = []

for i,j in enumerate(fileList):
    
    # Load each data set
    df = pd.read_csv(j, sep = '\t', nrows=29, names=['Variable', 'Value'])
    df = df.set_index('Variable').T # Transpose
    
    filePathList = os.path.normpath(j).split(os.path.sep)
    DataTrace = filePathList[-1]
    
    # Add the trace name to the dataframe
    df['DataTrace'] = DataTrace
    
    # Specific trace number from the string
    result = re.search('Trace_(.*)_Instron_Data.txt', DataTrace)
    
    # convert the string to an integer for sorting
    df['DataTrace#'] = int(result.group(1))
    
    finalAttr.append(df.tail(1).values.tolist())

# Compress list
finalAttr = [item for sublist in finalAttr for item in sublist]

# Create the new dataframe with the names from the previous data
df = pd.DataFrame(finalAttr, columns=df.columns.values)

# Experimental Data (ed)
ed = df.sort_values('DataTrace#').reset_index(drop=True)

#--- Save Data ---#
outputFileDirectory = os.path.join('Results', 'OutputFiles') # Folder

# Make folder if it doesn't exist
os.makedirs(outputFileDirectory, exist_ok=True)

# new File
outputFile = os.path.join(outputFileDirectory, 'PeelDataSummary.txt')

print('New file:', outputFile)

# Save results
ed.to_csv(outputFile, sep='\t', index=False, na_rep='nan')


# In[simulation results]
# Headers
names = ['FileName', 'Time', 'E1', 'E2', 'PT', 'G', 'V1', 'V2', 'R', 
         'F', 'MS', 'RE', 'VE', 'Knn', 'Kss', 'Ktt', 
         'DamageInitiation', 'tn', 'ts', 'tt', 'DamageEvolution', 
         'FE', 'Optimization', 'TIE', 'errorListL2Norm', 
         'ObjectiveFunction', 'SimSlope','SimMax', 'SimSS', 'simTime']

SF = os.path.join('Results', 'StatisticsFigures')

# Create folder if it doesn't exist
os.makedirs(SF, exist_ok=True)

# In[Elastic Modulus Convergende]

fp = os.path.join(cwd, 'Results', 'ElasticModulusConvergence')

fileList = glob.glob(os.path.join(fp, "sample*.txt"))

finalAttr = []
ElasticSummary = {} # Dictionary to look at each optimization routine

for i in fileList:
    
    # Load each data set
    df = pd.read_csv(i, sep = '\t')
    finalAttr.append(df.tail(1).values.tolist())

    # Append each data set to a single dictionary
    ElasticSummary[i] = df

# Compress list
finalAttr = [item for sublist in finalAttr for item in sublist]

# Create the new dataframe
df = pd.DataFrame(finalAttr, columns=names)

#--- Save Data ---#
outputFileDirectory = os.path.join('Results', 'OutputFiles') # Folder

# Make folder if it doesn't exist
os.makedirs(outputFileDirectory, exist_ok=True)

# new File
outputFile = os.path.join(outputFileDirectory, 'ElasticConvergenceSummary.txt')

print('New file:', outputFile)

# Save results
df.to_csv(outputFile, sep='\t', index=False, na_rep='nan')


Ev = df['VE'] # Elastic modulus

# In[Plots]

standardError = 68 # Used for confidence intervals

sns.set_theme(context='paper', style='darkgrid', palette="Paired", 
              font_scale=2)
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
custom_style = {'axes.facecolor': 'white', 
                'axes.edgecolor': 'black', 
                'axes.grid': False, 
                'axes.axisbelow': True, 
                'axes.labelcolor': 'black', 
                'figure.facecolor': 'white', 
                'grid.color': '.8', 
                'grid.linestyle': '-', 
                'text.color': 'black', 
                'xtick.color': 'black', 
                'ytick.color': 'black', 
                'xtick.direction': 'out', 
                'ytick.direction': 'out', 
                'lines.solid_capstyle': 'round', 
                'patch.edgecolor': 'w', 
                'patch.force_edgecolor': True, 
                'image.cmap': 'rocket', 
                'font.family': ['sans-serif'],
                'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 
                                    'Bitstream Vera Sans', 'sans-serif'], 
                'xtick.bottom': True, 
                'xtick.top': False, 
                'ytick.left': True, 
                'ytick.right': False, 
                'axes.spines.left': True, 
                'axes.spines.bottom': True, 
                'axes.spines.right': False, 
                'axes.spines.top': False, 
                'xtick.labelsize' : 16, 
                'ytick.labelsize' : 16, 
                'legend.title_fontsize' : 20}

# White background with ticks and black border lines, Turns grid off
ax = sns.set_style(rc=custom_style)

# In[Functions]

# fcn for plotting
def yfit(x):
    return slope*x + intercept

# In[Pivot table info]

pvtOut = {'count', np.median, np.mean, np.std} # pivot table outputs

# In[Plot simplifications]

R = 'Region'
Eq = 'Equator'
Po = 'Posterior'
AG = 'AgeGroup'
A60 = 'Age60'
Aleq60 = r'Age $\leq$ 60'
Ag60 = 'Age $>$ 60'
A = 'Age'

# Units
MPF = 'Maximum Peel Force (mN)'
SSPF = 'Steady-State Peel Force (mN)'
KDEUnit = r'Kernel Density Estimation'
ElasticUnit = r'Elastic Modulus (Pa)'
CohBehUnit = r'Cohesive Behavior (Pa)'
CohDMGUnit = r'Cohesive Damage Initiation (Pa)'
FEUnit = r'Fracture Energy (J)'

A_yrs = 'Age (yr.)'
A_G = 'Age Group (yr.)'


# In[Cohesive Behavior Convergence]

fp = os.path.join(cwd, 'Results', 'CohesiveBehaviorConvergence')

fileList = glob.glob(os.path.join(fp, "sample*.txt"))

finalAttr = []

for i,j in enumerate(fileList):
    
    # Load each data set
    df = pd.read_csv(j, sep = '\t')
    
    
    filePathList = os.path.normpath(j).split(os.path.sep)
    fileName = filePathList[-1]
    
    # Add the filename
    df['SimulationFileName'] = fileName
    
    # Specific trace number from the string
    result = re.search('Sample#(.*).txt', fileName)
    
    # convert the string to an integer for sorting
    df['DataTrace#'] = int(result.group(1))
    
    # Grab the final row (converged results)
    finalAttr.append(df.tail(1).values.tolist())

# Compress list
finalAttr = [item for sublist in finalAttr for item in sublist]

# Create the new dataframe
df = pd.DataFrame(finalAttr, columns=df.columns.values)

# Simulation Data (sd)
sd = df.sort_values('DataTrace#').reset_index(drop=True)

#--- Save Data ---#
outputFileDirectory = os.path.join('Results', 'OutputFiles') # Folder

# Make folder if it doesn't exist
os.makedirs(outputFileDirectory, exist_ok=True)

# new File
outputFile = os.path.join(outputFileDirectory, 'CohesiveBehaviorSummary.txt')

print('New file:', outputFile)

# Save results
sd.to_csv(outputFile, sep='\t', index=False, na_rep='nan')

Ev = sd['VE'] # Elastic modulus

# Cohesive Behaviour
sd['Knn'] = 2**sd['Knn'] # Elastic modulus
sd['Kss'] = 2**sd['Kss'] # Elastic modulus
sd['Ktt'] = 2**sd['Ktt'] # Elastic modulus

# Damage Initiation
sd['tn'] = 2**sd['tn'] # Normal 1
sd['ts'] = 2**sd['ts'] # Shear 1
sd['tt'] = 2**sd['tt'] # Shear 2

# Fracture energy
sd['FE'] = 2**sd['FE'] # Elastic modulus

# Outputs
errList = sd['errorListL2Norm']

SS = sd['SimSlope']
SM = sd['SimMax']
Ss = sd['SimSS']

# In[YouTube Links]

fp = os.path.join(cwd, 'Results', 'YouTube')

fileList = glob.glob(os.path.join(fp, "sample*.txt"))

finalAttr = []

for i,j in enumerate(fileList):
    
    # Load each data set
    df = pd.read_csv(j, names=['Link'])
    
    filePathList = os.path.normpath(j).split(os.path.sep)
    fileName = filePathList[-1]
    
    # Add the filename
    df['SimulationFileName'] = fileName
    
    # Specific trace number from the string
    result = re.search('Sample#(.*).txt', fileName)
    
    tNum = int(result.group(1)) # Trace number
    
    # convert the string to an integer for sorting
    df['DataTrace#'] = tNum
    
    # Hyperlink for LaTeX
    df['HyperLink'] = ('\href{' + '{}'.format(df['Link'][0]) + 
                       '}{' + '{}'.format(tNum) + '}')
    
    # Grab the final row (converged results)
    finalAttr.append(df.tail(1).values.tolist())

# Compress list
finalAttr = [item for sublist in finalAttr for item in sublist]

# Create the new dataframe
df = pd.DataFrame(finalAttr, columns=df.columns.values)

# YouTube (yt)
yt = df.sort_values('DataTrace#').reset_index(drop=True)

#--- Save Data ---#
outputFileDirectory = os.path.join('Results', 'OutputFiles') # Folder

# Make folder if it doesn't exist
os.makedirs(outputFileDirectory, exist_ok=True)

# new File
outputFile = os.path.join(outputFileDirectory, 'YouTube.txt')

print('New file:', outputFile)

# In[Cohesive Behavior Convergence]

fp = os.path.join(cwd, 'Results', 'CohesiveBehaviorConvergence')

fileList = glob.glob(os.path.join(fp, "sample*.txt"))

finalAttr = []

CohesiveSummary = {}

for i,j in enumerate(fileList):
    
    # Load each data set
    df = pd.read_csv(j, sep = '\t')
    
    
    filePathList = os.path.normpath(j).split(os.path.sep)
    fileName = filePathList[-1]
    
    # Add the filename
    df['SimulationFileName'] = fileName
    
    # Specific trace number from the string
    result = re.search('Sample#(.*).txt', fileName)
    
    # convert the string to an integer for sorting
    df['DataTrace#'] = int(result.group(1))
    
    # Grab the final row (converged results)
    finalAttr.append(df.tail(1).values.tolist())
    
    # Append each data set to a single dictionary
    CohesiveSummary[i] = df

# Compress list
finalAttr = [item for sublist in finalAttr for item in sublist]

# Create the new dataframe
df = pd.DataFrame(finalAttr, columns=df.columns.values)

# Simulation Data (sd)
sd = df.sort_values('DataTrace#').reset_index(drop=True)

#--- Save Data ---#
outputFileDirectory = os.path.join('Results', 'OutputFiles') # Folder

# Make folder if it doesn't exist
os.makedirs(outputFileDirectory, exist_ok=True)

# new File
outputFile = os.path.join(outputFileDirectory, 'CohesiveBehaviorSummary.txt')

print('New file:', outputFile)

# In[Merge data sets (Experimental & Simulation)]

# Merge experimental and simulation data
md1 = pd.merge(sd, ed, on='DataTrace#')
md = pd.merge(md1, yt, on='DataTrace#')

# Simplifications
R = 'Region'
Eq = 'Equator'
Po = 'Posterior'
A60 = 'Age60'
Aleq60 = r'Age $\leq$ 60'
Ag60 = r'Age $>$ 60'
A = 'Age'

# Redo some columns for plotting
md[A] = md['Human Age:']
md[R] = md['Human Region:']

# Break age groups into bins
bins = [0, 60, 90]
labelsAge60 = [Aleq60, Ag60]

# Properly update parameters
# Cohesive Behaviour
md['Knn'] = 2**md['Knn'] # Elastic modulus
md['Kss'] = 2**md['Kss'] # Elastic modulus
md['Ktt'] = 2**md['Ktt'] # Elastic modulus

# Damage Initiation
md['tn'] = 2**md['tn'] # Normal 1
md['ts'] = 2**md['ts'] # Shear 1
md['tt'] = 2**md['tt'] # Shear 2

# Fracture energy
md['FE'] = 2**md['FE'] # Elastic modulus

# Create binned AgeGroups
md[A60] = pd.cut(md[A].astype(int), bins, labels=labelsAge60, right=True)

# Convert Strings to floats/integers
md['EV'] = pd.to_numeric(md['VE'], downcast="float")

md['Trace'] = pd.to_numeric(md['DataTrace#'], downcast="integer")
md['$Exp_\max$'] = pd.to_numeric(md['FMax (mN):'], downcast="float")
md['$Exp_{SS}$'] = pd.to_numeric(md['FSS (mN):'], downcast="float")
md['$Sim_\max$'] = pd.to_numeric(md['SimMax'], downcast="float")
md['$Sim_{SS}$'] = pd.to_numeric(md['SimSS'], downcast="float")
md['$L^2$ Norm'] = pd.to_numeric(md['errorListL2Norm'], downcast="float")
md[A] = pd.to_numeric(md[A], downcast="integer")

# Simplify for later
md['L/R'] = np.where(md['Human Left/Right:'] == 'Left', 'L', 'R')
md['Region'] = np.where(md['Region'] == 'Equator', 'Eq.', 'Po.')

#--- Save Data ---#
outputFileDirectory = os.path.join('Results', 'OutputFiles') # Folder

# Make folder if it doesn't exist
os.makedirs(outputFileDirectory, exist_ok=True)

# new File
outputFile = os.path.join(outputFileDirectory, 'ExpSimSummary.txt')

print('New file:', outputFile)

# Save results
md.to_csv(outputFile, sep='\t', index=False, na_rep='nan')

# Create specific LaTeX table

# Add the index groups and convert NaN's to "-"'s
tabColumns = ['Trace', 
              A, 
              'L/R', 
              'Region', 
              '$Exp_\max$', 
              '$Exp_{SS}$', 
              '$Sim_\max$', 
              '$Sim_{SS}$', 
              '$L^2$ Norm', 
              'HyperLink']

print(md.to_latex(index=False, columns=tabColumns, na_rep='-', escape=False, 
                  float_format="{:0.3f}".format))

# In[Add full name to region after the table was created]

# md[R] = np.where(md[R] == 'Eq.', 'Equator', 'Posterior') # Difficult to use
md.loc[md[R] == 'Eq.', R] = Eq
md.loc[md[R] == 'Po.', R] = Po

# In[Smart Plot]

def boxPlotBlackBorder(ax):
    # iterate over boxes in the plot to make each line black
    for i,box in enumerate(ax.artists):
        box.set_edgecolor('black')
        # box.set_facecolor('white')
    
        # iterate over whiskers and median lines
        for j in range(6*i, 6*(i+1)):
            ax.lines[j].set_color('black')

def smartPlot(data=None, x=None, y=None, hue=None, hue_order=None, 
              addBoxPair=None, ci=None, errcolor=None, capsize=None, 
              plot=None, test=None, sigLoc=None, text_format=None, 
              line_offset=None, line_offset_to_box=None, line_height=None, 
              fontsize=None, legLoc=None, verbose=None, yAxis=None, 
              xlabel=None, ylabel=None, legendTitle=None, figName=None, 
              folderName=None, dataPoints=None, stats=None):
    
    # barplot
    scale = 1.6
    base = 10
    f, ax = plt.subplots(figsize=(base*scale, base))

    if plot == 'barplot':
        ax = sns.barplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, 
                         ci=ci, errcolor=errcolor, capsize=capsize)
    
    elif plot == 'boxplot':
        ax = sns.boxplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order)
    
    # Statistical test for differences
    x_grps = list(data[x].unique()) # List of groups
    if hue != None:
        # Create combinations to compare
        box_pairs_1 = [((x_grps_i, hue_order[0]), 
                        (x_grps_i, hue_order[1])) 
                       for x_grps_i in x_grps]
        box_pairs = box_pairs_1
        
        if addBoxPair != None:
            # Additional box pairs
            box_pairs =  box_pairs_1 + addBoxPair
    
    elif hue_order != None:
        box_pairs = [(hue_order[0], hue_order[1])]
    
    if yAxis != None:
            ax.set_yscale("log")
    
    if stats != None:
        #Stats results and significant differences (SR)
        SR = add_stat_annotation(ax, plot=plot, data=data, x=x, y=y, hue=hue, 
                                 hue_order=hue_order, box_pairs=box_pairs, 
                                 test=test, loc=sigLoc, 
                                 text_format=text_format, verbose=verbose, 
                                 comparisons_correction=None, 
                                 line_offset=line_offset, 
                                 line_offset_to_box=line_offset_to_box, 
                                 line_height= line_height, 
                                 fontsize=fontsize) # 'bonferroni'
    
    if plot == 'boxplot':
        boxPlotBlackBorder(ax) # Make borders black
    
    if dataPoints == True:
        # Add data points to the box plot
        sns.stripplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, 
                      color='.5', size=5, linewidth=1, dodge=True)
        
        # gather plot attributes for legends
        handles, labels = ax.get_legend_handles_labels()
        
        if hue != None:
            l = plt.legend(handles[0:2], labels[0:2], title=legendTitle, 
                           fontsize=18)
    
    else:
        if hue != None:
            ax.legend(loc=legLoc, fontsize=18).set_title(legendTitle)
    
    if hue != None and hue_order != None:
        # for legend title
        plt.setp(ax.get_legend().get_title(), fontsize=18)
        
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    
    # Adjust fonts, because it doesn't seem to work
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
        
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    
    ax = sns.despine() # takes the lines off on the right and top of the graph
    
    if folderName != None:
        # If a new folder name is given, put the files there
        
        # New file path
        NP = os.path.join(SF, folderName)
        
        # Create folder if it doesn't exist
        os.makedirs(NP, exist_ok=True)
        
    else:
        # Put the file in the same folder
        NP = SF
    
    f.savefig(os.path.join(NP, '{}.pdf'.format(figName)), 
              bbox_inches='tight')
    plt.close()

def twoWayAnova(data=None, var=None, A=None, B=None, fileName=None, 
                filePath=None):
    """
    Two-Way Anova
    Parameters
    ----------
    data : dataframe
    var : continuous variable
    A : Effect #1
    B : Effect #2
    fileName : Filename
    filePath : filepath
    """
    model = ols(f'{var} ~ {A} + {B} + {A}:{B}', data=data).fit()
    
    res = sm.stats.anova_lm(model, typ=2)
    
    print(80*'-', 2*'\n', 'Two-way ANOVA\n', res, 2*'\n')
    
    NP = os.path.join(SF, filePath)
    
    # Create folder if it doesn't exist
    os.makedirs(NP, exist_ok=True)
    
    f = open(os.path.join(NP, f'{fileName}.txt'), "w")
    f.write(var + '\twith effects:  ' + A + ' and ' + B)
    f.write('\n')
    f.write(res.to_string())
    f.close()

# In[Elastic modulus group plots by region and age +/- 60]
EV = 'EV'
Ev = [EV]
Folder = 'ElasticModulus'

# Additional group
addBoxPair1 = [((Aleq60, Eq), (Age_Group_i, Eq)) 
               for Age_Group_i in list(md[A60].unique())]

addBoxPair2 = [((Aleq60, Eq), (Ag60, Po))]

addBoxPair = addBoxPair1 #+ addBoxPair2

for i in Ev:
    pivotEv = pd.pivot_table(md, values=i, index=[A60, R], 
                                                 aggfunc=pvtOut)
    
    print('pivotEv')
    print(pivotEv)
    # Add the index groups and convert NaN's to "-"'s
    print(pivotEv.to_latex(index=True, na_rep='-', escape=False, 
                           float_format="{:0.3f}".format))

    # Barplot
    smartPlot(data=md, x=A60, y=i, hue=R, hue_order=[Eq, Po], ci=68, 
              errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
              sigLoc='outside', text_format='star', line_offset=0.015, 
              line_offset_to_box=0.0, line_height=0.015, fontsize=16, 
              legLoc='best', verbose=2, yAxis=None, 
              xlabel=A_G, ylabel=ElasticUnit, legendTitle=R, 
              figName='RegionAge_BarPlot', folderName=Folder, 
              addBoxPair=addBoxPair, stats=None)
    
    # Boxplot
    smartPlot(data=md, x=A60, y=i, hue=R, hue_order=[Eq, Po], plot='boxplot', 
              test='t-test_ind', sigLoc='outside', text_format='star', 
              line_offset=0.015, line_offset_to_box=0.0, line_height=0.015, 
              fontsize=16, legLoc='best', verbose=2, yAxis=None, 
              xlabel=A_G, ylabel=ElasticUnit, addBoxPair=addBoxPair, 
              legendTitle=R, figName='RegionAge_BoxPlot', folderName=Folder, 
              stats=None)
    
    # Boxplot with data
    smartPlot(data=md, x=A60, y=i, hue=R, hue_order=[Eq, Po], plot='boxplot', 
              test='t-test_ind', sigLoc='outside', text_format='star', 
              line_offset=0.015, line_offset_to_box=0.0, line_height=0.015, 
              fontsize=16, legLoc='best', verbose=2, yAxis=None, 
              xlabel=A_G, ylabel=ElasticUnit, addBoxPair=addBoxPair, 
              legendTitle=R, 
              figName='RegionAge_BoxPlotWithData', folderName=Folder, 
              dataPoints=True, stats=None)

twoWayAnova(data=md, var='VE', A='Age', B='Region', 
            fileName='Age_Region_2wayAnova', 
            filePath=Folder)

twoWayAnova(data=md, var='VE', A=A60, B=R, 
            fileName='AgeGroup_Region_2wayAnova', 
            filePath=Folder)

# In[Elastic modulus group plots by Age Group +/- 60]
EV = 'EV'
Ev = [EV]
Folder = 'ElasticModulus'

# Additional group
addBoxPair1 = [(Aleq60, Ag60)]

addBoxPair = addBoxPair1

# Create a matching column for repeated measures
md['MatchingID'] = md['Human ID:'].map(str) + md['Human Region:'].map(str)

# matched_pairs student's t-test

# dfMP = md[md.duplicated(['MatchingID'], keep=False)]
# f, p = stats.ttest_rel(dfMP[Ev][dfMP[R] == Eq], 
#                        dfMP[Ev][dfMP[R] == Po])

data = md # dfMP

for i in Ev:
    pivotEv = pd.pivot_table(md, values=i, index=[A60, R], 
                                                 aggfunc=pvtOut)
    
    print('pivotEv')
    print(pivotEv)
    # Add the index groups and convert NaN's to "-"'s
    print(pivotEv.to_latex(index=True, na_rep='-', escape=False, 
                           float_format="{:0.3f}".format))
    
    # Barplot
    smartPlot(data=data, x=A60, y=i, hue=None, hue_order=[Aleq60, Ag60], ci=68, 
              errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
              sigLoc='outside', text_format='star', line_offset=0.015, 
              line_offset_to_box=0.0, line_height=0.015, fontsize=16, 
              legLoc='best', verbose=2, yAxis=None, 
              xlabel=A_G, ylabel=ElasticUnit, legendTitle=None, 
              figName='Age_BarPlot', folderName=Folder, 
              addBoxPair=addBoxPair, stats=True)
    
    # Boxplot
    smartPlot(data=data, x=A60, y=i, hue=None, hue_order=[Aleq60, Ag60],
              plot='boxplot', 
              test='t-test_ind', sigLoc='outside', text_format='star', 
              line_offset=0.015, line_offset_to_box=0.0, line_height=0.015, 
              fontsize=16, legLoc='best', verbose=2, yAxis=None, 
              xlabel=A_G, ylabel=ElasticUnit, addBoxPair=addBoxPair, 
              legendTitle=R, figName='Age_BoxPlot', folderName=Folder, 
              stats=True)
    
    # Boxplot with data
    smartPlot(data=data, x=A60, y=i, hue=None, hue_order=[Aleq60, Ag60], 
              plot='boxplot', 
              test='t-test_ind', sigLoc='outside', text_format='star', 
              line_offset=0.015, line_offset_to_box=0.0, line_height=0.015, 
              fontsize=16, legLoc='best', verbose=2, yAxis=None, 
              xlabel=A_G, ylabel=ElasticUnit, addBoxPair=addBoxPair, 
              legendTitle=R, 
              figName='Age_BoxPlotWithData', folderName=Folder, 
              dataPoints=True, stats=True)

# In[Elastic modulus group plots by region]
EV = 'EV'
Ev = [EV]
Folder = 'ElasticModulus'

# Additional group
addBoxPair1 = [(Eq, Po)]

addBoxPair = addBoxPair1

for i in Ev:
    pivotEv = pd.pivot_table(md, values=i, index=[A60, R], 
                                                 aggfunc=pvtOut)
    
    print('pivotEv')
    print(pivotEv)
    # Add the index groups and convert NaN's to "-"'s
    print(pivotEv.to_latex(index=True, na_rep='-', escape=False, 
                           float_format="{:0.3f}".format))
    
    # Barplot
    smartPlot(data=md, x=R, y=i, hue=None, hue_order=[Eq, Po], ci=68, 
              errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
              sigLoc='outside', text_format='star', line_offset=0.015, 
              line_offset_to_box=0.0, line_height=0.015, fontsize=16, 
              legLoc='best', verbose=2, yAxis=None, 
              xlabel=A_G, ylabel=ElasticUnit, legendTitle=None, 
              figName='Age_BarPlot', folderName=Folder, 
              addBoxPair=addBoxPair, stats=True)
    
    # Boxplot
    smartPlot(data=md, x=R, y=i, hue=None, hue_order=[Eq, Po], plot='boxplot', 
              test='t-test_ind', sigLoc='outside', text_format='star', 
              line_offset=0.015, line_offset_to_box=0.0, line_height=0.015, 
              fontsize=16, legLoc='best', verbose=2, yAxis=None, 
              xlabel=A_G, ylabel=ElasticUnit, addBoxPair=addBoxPair, 
              legendTitle=R, figName='Age_BoxPlot', folderName=Folder, 
              stats=True)
    
    # Boxplot with data
    smartPlot(data=md, x=R, y=i, hue=None, hue_order=[Eq, Po], plot='boxplot', 
              test='t-test_ind', sigLoc='outside', text_format='star', 
              line_offset=0.015, line_offset_to_box=0.0, line_height=0.015, 
              fontsize=16, legLoc='best', verbose=2, yAxis=None, 
              xlabel=A_G, ylabel=ElasticUnit, addBoxPair=addBoxPair, 
              legendTitle=R, 
              figName='Age_BoxPlotWithData', folderName=Folder, 
              dataPoints=True, stats=True)

# In[Elastic Modulus Regression by Age in both Regions]

Folder = 'ElasticModulus'

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
ax = sns.lmplot(data=md, x=A, y=EV, hue=R, markers=["o", "x"], 
                legend_out=False, fit_reg=True, height=5, aspect=1.6, 
                palette="Set1", truncate=True, ci=95, line_kws={'lw':0})

ax.set(xlabel=A_yrs, ylabel=ElasticUnit)

ax = ax.axes[0][0] # Convert faceted grid to matplotlib fig

# Remove all NaN's from the data for regressions
# remove nans from ILM thickness & Max
df_no_Nan = md.dropna(subset=[A, EV])

# linear regressions for fitting
x = df_no_Nan[A][df_no_Nan[R] == Eq]
# Convert to N
y = df_no_Nan[EV][df_no_Nan[R] == Eq]

x_plot = np.linspace(min(x), max(x), 100)

slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1, label='line')
ax.text(62, yfit(62) + 5, r'$r={:.4f}$'.format(r_value1), color='r', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'Elastic Modulus and Age in the Equator\n', 
      f'P={p_value:.4f}', f'r={r_value1:.4f}')

# linear regressions for fitting
x = df_no_Nan[A][df_no_Nan[R] == Po]
y = df_no_Nan[EV][df_no_Nan[R] == Po]

x_plot = np.linspace(min(x), max(x), 100)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1, label='line')
ax.text(62, yfit(62) + 5, r'$r={:.4f}$'.format(r_value2), color='b', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'Elastic Modulus and Age in the Equator\n', 
      f'P={p_value:.4f}', f'r={r_value2:.4f}')

# Axis limits
ax.set(ylim=(0, 500))
ax.set(xlim=(29, 80))

# New path
NP = os.path.join(SF, Folder)

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

plt.savefig(os.path.join(NP, 'Regression_Age_by_Region.pdf'), 
           bbox_inches='tight')
plt.close()


# In[Elastic Modulus Regression by Max Peel Force in both regions]

Fmax = 'SimMax'
Folder = 'ElasticModulus'

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
ax = sns.lmplot(data=md, x=EV, y=Fmax, hue=R, hue_order= [Eq, Po], 
                markers=["o", "x"], legend_out=False, fit_reg=True, 
                height=5, aspect=1.6, palette="Set1", truncate=True, ci=95, 
                line_kws={'lw':0})

ax.set(xlabel=ElasticUnit, ylabel=MPF)

ax = ax.axes[0][0] # Convert faceted grid to matplotlib fig

# Remove all NaN's from the data for regressions
# remove nans from ILM thickness & Max
df_no_Nan = md.dropna(subset=[Fmax, EV])

# linear regressions for fitting
x = df_no_Nan[EV][df_no_Nan[R] == Eq]
# Convert to N
y = df_no_Nan[Fmax][df_no_Nan[R] == Eq]

x_plot = np.linspace(min(x), max(x), 100)

slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1, label='line')
ax.text(400, yfit(400) - 1, r'$r={:.4f}$'.format(r_value1), color='r', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'Elastic Modulus and Max Force in the Equator\n', 
      f'P={p_value:.4f}', f'r={r_value1:.4f}')

# linear regressions for fitting
x = df_no_Nan[EV][df_no_Nan[R] == Po]
y = df_no_Nan[Fmax][df_no_Nan[R] == Po]

x_plot = np.linspace(min(x), max(x), 100)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1, label='line')
ax.text(220, yfit(220) - 1, r'$r={:.4f}$'.format(r_value2), color='b', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'Elastic Modulus and Max Force in the Equator\n', 
      f'P={p_value:.4f}', f'r={r_value2:.4f}')

# Axis limits
ax.set(ylim=(0, 18))
ax.set(xlim=(0, 500))

# New path
NP = os.path.join(SF, Folder)

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

plt.savefig(os.path.join(NP, 'Regression_MaxForce_by_Region.pdf'), 
           bbox_inches='tight')
plt.close()


# In[Elastic Modulus Regression by Steady-State Peel Force in both regions]

FSS = 'SimSS'

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
ax = sns.lmplot(data=md, x=EV, y=FSS, hue=R, hue_order=[Eq, Po], 
                markers=["o", "x"], legend_out=False, fit_reg=True, 
                height=5, aspect=1.6, palette="Set1", truncate=True, ci=95, 
                line_kws={'lw':0})

ax.set(xlabel=ElasticUnit, ylabel=SSPF)

ax = ax.axes[0][0] # Convert faceted grid to matplotlib fig

# Remove all NaN's from the data for regressions
# remove nans from ILM thickness & Max
df_no_Nan = md.dropna(subset=[FSS, EV])

# linear regressions for fitting
x = df_no_Nan[EV][df_no_Nan[R] == Eq]
# Convert to N
y = df_no_Nan[FSS][df_no_Nan[R] == Eq]

x_plot = np.linspace(min(x), max(x), 100)

slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1, label='line')
ax.text(400, yfit(400) - 1, r'$r={:.4f}$'.format(r_value1), color='r', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'Elastic Modulus and SS Force in the Equator\n', 
      f'P={p_value:.4f}', f'r={r_value1:.4f}')

# linear regressions for fitting
x = df_no_Nan[EV][df_no_Nan[R] == Po]
y = df_no_Nan[FSS][df_no_Nan[R] == Po]

x_plot = np.linspace(min(x), max(x), 100)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1, label='line')
ax.text(220, yfit(220) - 1, r'$r={:.4f}$'.format(r_value2), color='b', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'Elastic Modulus and SS Force in the Equator\n', 
      f'P={p_value:.4f}', f'r={r_value2:.4f}')

# Axis limits
ax.set(ylim=(0, 12))
ax.set(xlim=(0, 500))

# New path
NP = os.path.join(SF, 'ElasticModulus')

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

plt.savefig(os.path.join(NP, 'Regression_SSForce_by_Region.pdf'), 
           bbox_inches='tight')
plt.close()

# In[Cohesive parameter group plots]

Knn = 'Knn'
Kss = 'Kss'
Ktt = 'Ktt'

Folder = 'CohesiveBehavior'

# Filter data (Brittany)
dfKnn = md[md[Knn] > 25e6]
dfKss = md[md[Kss] > 100e6]
dfKtt = md[md[Ktt] > 150e6]

dfFilt = {Knn: dfKnn, 
          Kss: dfKss, 
          Ktt: dfKtt}

for key, val in dfFilt.items():
    pivotCohBeh = pd.pivot_table(val, values=key, index=[A60, R], 
                                                 aggfunc=pvtOut)
    
    print('pivotCohBeh')
    print(pivotCohBeh)
    # Add the index groups and convert NaN's to "-"'s
    print(pivotCohBeh.to_latex(index=True, na_rep='-', escape=False, 
                               float_format="{:0.3f}".format))

    # Barplot
    smartPlot(data=val, x=A60, y=key, hue=R, hue_order=[Eq, Po], ci=68, 
              errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
              sigLoc='outside', text_format='star', line_offset=0.015, 
              line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
              legLoc='best', verbose=2, yAxis='log', 
              xlabel=A_G, ylabel=CohBehUnit, legendTitle=R, 
              figName=f'Region_BarPlot_{key}', folderName=Folder)
    
    # Boxplot
    smartPlot(data=val, x=A60, y=key, hue=R, hue_order=[Eq, Po],
              plot='boxplot', 
              test='t-test_ind', sigLoc='outside', text_format='star', 
              line_offset=0.015, line_offset_to_box=0.0, line_height=0.015, 
              fontsize='small', legLoc='best', verbose=2, yAxis='log', 
              xlabel=A_G, ylabel=CohBehUnit, 
              legendTitle=R, figName=f'Region_BoxPlot_{key}', 
              folderName=Folder)
    
    # Boxplot with data
    smartPlot(data=val, x=A60, y=key, hue=R, hue_order=[Eq, Po], 
              plot='boxplot', 
              test='t-test_ind', sigLoc='outside', text_format='star', 
              line_offset=0.015, line_offset_to_box=0.0, line_height=0.015, 
              fontsize='small', legLoc='best', verbose=2, yAxis='log', 
              xlabel=A_G, ylabel=CohBehUnit, 
              legendTitle=R, 
              figName=f'Region_BoxPlotWithData_{key}', folderName=Folder, 
              dataPoints=True)

# In[Kss Regression by Max Peel Force in both regions]

Fmax = 'SimMax'
Folder = 'CohesiveBehavior'

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
ax = sns.lmplot(data=dfKnn, x='Kss', y=Fmax, hue=R, hue_order= [Eq, Po], 
                markers=["o", "x"], legend_out=False, fit_reg=True, 
                height=5, aspect=1.6, palette="Set1", truncate=True, ci=95, 
                line_kws={'lw':0})

ax.set(xscale="log")

ax.set(xlabel=CohBehUnit, ylabel=MPF)

ax = ax.axes[0][0] # Convert faceted grid to matplotlib fig

# Remove all NaN's from the data for regressions
# remove nans from ILM thickness & Max
df_no_Nan = dfKnn.dropna(subset=[Fmax, 'Kss'])

# linear regressions for fitting
x = df_no_Nan['Kss'][df_no_Nan[R] == Eq]
# Convert to N
y = df_no_Nan[Fmax][df_no_Nan[R] == Eq]

x_plot = np.linspace(min(x), max(x), 100)

slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1, label='line')
ax.text(5e7, yfit(5e7) + 1, r'$r={:.4f}$'.format(r_value1), color='r', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'Kss and Max Force in the Equator\n', 
      f'P={p_value:.4f}', f'r={r_value1:.4f}')

# linear regressions for fitting
x = df_no_Nan['Kss'][df_no_Nan[R] == Po]
y = df_no_Nan[Fmax][df_no_Nan[R] == Po]

x_plot = np.linspace(min(x), max(x), 100)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1, label='line')
ax.text(6e7, yfit(6e7) + 1, r'$r={:.4f}$'.format(r_value2), color='b', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'Kss and Max Force in the Equator\n', 
      f'P={p_value:.4f}', f'r={r_value2:.4f}')

# Axis limits
# ax.set(ylim=(0, 18))
ax.set(xlim=(3e7, 9e7))

# New path
NP = os.path.join(SF, Folder)

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

plt.savefig(os.path.join(NP, 'Kss_vs_MaxForce_by_Region.pdf'), 
           bbox_inches='tight')
plt.close()

# In[Kss Regression by Steady State Peel Force in both regions]

FSS = 'SimSS'
Folder = 'CohesiveBehavior'

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
ax = sns.lmplot(data=dfKnn, x='Kss', y=FSS, hue=R, hue_order= [Eq, Po], 
                markers=["o", "x"], legend_out=False, fit_reg=True, 
                height=5, aspect=1.6, palette="Set1", truncate=True, ci=95, 
                line_kws={'lw':0})

ax.set(xscale="log")

ax.set(xlabel=CohBehUnit, ylabel=SSPF)

ax = ax.axes[0][0] # Convert faceted grid to matplotlib fig

# Remove all NaN's from the data for regressions
# remove nans from ILM thickness & Max
df_no_Nan = dfKnn.dropna(subset=[FSS, 'Kss'])

# linear regressions for fitting
x = df_no_Nan['Kss'][df_no_Nan[R] == Eq]
# Convert to N
y = df_no_Nan[FSS][df_no_Nan[R] == Eq]

x_plot = np.linspace(min(x), max(x), 100)

slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1, label='line')
ax.text(5e7, yfit(5e7) + 1, r'$r={:.4f}$'.format(r_value1), color='r', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'Kss and Steady State Peel Force in the Equator\n', 
      f'P={p_value:.4f}', f'r={r_value1:.4f}')

# linear regressions for fitting
x = df_no_Nan['Kss'][df_no_Nan[R] == Po]
y = df_no_Nan[FSS][df_no_Nan[R] == Po]

x_plot = np.linspace(min(x), max(x), 100)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1, label='line')
ax.text(6e7, yfit(6e7) + 1, r'$r={:.4f}$'.format(r_value2), color='b', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'Kss and Steady State Peel Force in the Equator\n', 
      f'P={p_value:.4f}', f'r={r_value2:.4f}')

# Axis limits
# ax.set(ylim=(0, 18))
ax.set(xlim=(3e7, 9e7))

# New path
NP = os.path.join(SF, Folder)

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

plt.savefig(os.path.join(NP, 'Kss_vs_SSForce_by_Region.pdf'), 
           bbox_inches='tight')
plt.close()

# In[Cohesive Damage Initiation parameter group plots]

tn = 'tn'
ts = 'ts'
tt = 'tt'

Folder = 'CohesiveDamage'

# Filter data (Brittany)
dftn = md[md[tn] < 3000]
dfts = md[md[ts] < 3000]
dftt = md[md[tt] < 3000]

dfFilt = {tn: dftn, 
          ts: dfts, 
          tt: dftt}

for key, val in dfFilt.items():
    pivotCohDMG = pd.pivot_table(val, values=key, index=[A60, R], 
                                                 aggfunc=pvtOut)
    
    print('pivotCohDMG')
    print(pivotCohDMG)
    # Add the index groups and convert NaN's to "-"'s
    print(pivotCohDMG.to_latex(index=True, na_rep='-', escape=False, 
                               float_format="{:0.3f}".format))
    
    # Barplot
    smartPlot(data=val, x=A60, y=key, hue=R, hue_order=[Eq, Po], ci=68, 
              errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
              sigLoc='outside', text_format='star', line_offset=0.015, 
              line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
              legLoc='best', verbose=2, yAxis=None, 
              xlabel=A_G, ylabel=CohDMGUnit, legendTitle=R, 
              figName=f'Region_BarPlot_{key}', folderName=Folder)
    
    # Boxplot
    smartPlot(data=val, x=A60, y=key, hue=R, hue_order=[Eq, Po], 
              plot='boxplot', test='t-test_ind', sigLoc='outside', 
              text_format='star', line_offset=0.015, line_offset_to_box=0.0, 
              line_height=0.015, 
              fontsize='small', legLoc='best', verbose=2, yAxis=None, 
              xlabel=A_G, ylabel=CohDMGUnit, 
              legendTitle=R, figName=f'Region_BoxPlot_{key}', folderName=Folder)
    
    # Boxplot with data
    smartPlot(data=val, x=A60, y=key, hue=R, hue_order=[Eq, Po], 
              plot='boxplot', test='t-test_ind', sigLoc='outside',
              text_format='star', line_offset=0.015, line_offset_to_box=0.0, 
              line_height=0.015, 
              fontsize='small', legLoc='best', verbose=2, yAxis=None, 
              xlabel=A_G, ylabel=CohDMGUnit, 
              legendTitle=R, 
              figName=f'Region_BoxPlotWithData_{key}', folderName=Folder, 
              dataPoints=True)

# In[ts Regression by Max Peel Force in both regions]


mod = sm.OLS(df['VE'], df[Fmax])
res = mod.fit()
print(80*'-', 2*'\n', 'Correlation between Age & E\n', res.summary())

text_file = open('Correlation.txt', "w")
text_file.write(res.summary().as_text())
text_file.close()


stats.ttest_ind(df['VE'], df[Fmax])


Fmax = 'SimMax'
Folder = 'CohesiveDamage'

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
ax = sns.lmplot(data=dfts, x=ts, y=Fmax, hue=R, hue_order= [Eq, Po], 
                markers=["o", "x"], legend_out=False, fit_reg=True, 
                height=5, aspect=1.6, palette="Set1", truncate=True, ci=95, 
                line_kws={'lw':0})

# ax.set(xscale="log")

ax.set(xlabel=CohDMGUnit, ylabel=MPF)

ax = ax.axes[0][0] # Convert faceted grid to matplotlib fig

# Remove all NaN's from the data for regressions
# remove nans from ILM thickness & Max
df_no_Nan = dfts.dropna(subset=[Fmax, ts])

# linear regressions for fitting
x = df_no_Nan[ts][df_no_Nan[R] == Eq]
# Convert to N
y = df_no_Nan[Fmax][df_no_Nan[R] == Eq]

x_plot = np.linspace(min(x), max(x), 100)

xt = 3*10**2 # Location of text

slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1, label='line')
plt.text(xt, yfit(xt) + 1, r'$r={:.4f}$'.format(r_value1), color='r', 
          horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'ts and Max Force in the Equator\n', 
      f'P={p_value:.4f}', f'r={r_value1:.4f}')

# linear regressions for fitting
x = df_no_Nan[ts][df_no_Nan[R] == Po]
y = df_no_Nan[Fmax][df_no_Nan[R] == Po]

xt = 8*10**2 # Location of text

x_plot = np.linspace(min(x), max(x), 100)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1, label='line')
plt.text(xt, yfit(xt) + 1, r'$r={:.4f}$'.format(r_value2), color='b', 
          horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'ts and Max Force in the Equator\n', 
      f'P={p_value:.4f}'.format(), f'r={r_value2:.4f}')

# Axis limits
# ax.set(ylim=(0, 18))
# ax.set(xlim=(3e7, 9e7))
ax.set(xlim=(1.8*10**2, 1.3*10**3))

# New path
NP = os.path.join(SF, Folder)

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

plt.savefig(os.path.join(NP, 'ts_vs_MaxForce_by_Region.pdf'), 
           bbox_inches='tight')
plt.close()

# In[tt Regression by Max Peel Force in both regions]

Fmax = 'SimMax'
Folder = 'CohesiveDamage'

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
ax = sns.lmplot(data=dftt, x=tt, y=Fmax, hue=R, hue_order= [Eq, Po], 
                markers=["o", "x"], legend_out=False, fit_reg=True, 
                height=5, aspect=1.6, palette="Set1", truncate=True, ci=95, 
                line_kws={'lw':0})

# ax.set(xscale="log")

ax.set(xlabel=CohDMGUnit, ylabel=MPF)

ax = ax.axes[0][0] # Convert faceted grid to matplotlib fig

# Remove all NaN's from the data for regressions
# remove nans from ILM thickness & Max
df_no_Nan = dftt.dropna(subset=[Fmax, tt])

# linear regressions for fitting
x = df_no_Nan[tt][df_no_Nan[R] == Eq]
# Convert to N
y = df_no_Nan[Fmax][df_no_Nan[R] == Eq]

x_plot = np.linspace(min(x), max(x), 100)

xt = 2*10**2 # Location of text

slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1, label='line')
ax.text(xt, yfit(xt) + 2, r'$r={:.4f}$'.format(r_value1), color='r', 
          horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'tt and Max Force in the Equator\n', 
      f'P={p_value:.4f}', f'r={r_value1:.4f}')

# linear regressions for fitting
x = df_no_Nan[tt][df_no_Nan[R] == Po]
y = df_no_Nan[Fmax][df_no_Nan[R] == Po]

xt = 1.2*10**3 # Location of text

x_plot = np.linspace(min(x), max(x), 100)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)
ax.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1, label='line')
ax.text(xt, yfit(xt) + 1, r'$r={:.4f}$'.format(r_value2), color='b', 
          horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'tt and Max Force in the Equator\n', 
      f'P={p_value:.4f}', f'r={r_value2:.4f}')

# Axis limits
ax.set(ylim=(0, None))
ax.set(xlim=(1.*10**1, 2*10**3))

# New path
NP = os.path.join(SF, Folder)

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

plt.savefig(os.path.join(NP, 'tt_vs_MaxForce_by_Region.pdf'), 
           bbox_inches='tight')
plt.close()

# In[tt Regression by Age regions]

Folder = 'CohesiveDamage'

Aleq63 = r'Age $\leq$ 63'
Ag63 = 'Age $>$ 63'

bins = [0, 63, 90]
labelsAge63 = [Aleq63, Ag63]

# Create binned AgeGroups
A63 = 'A63'
md[A63] = pd.cut(md[A], bins, labels=labelsAge63, right=True)

dftt = md[md[tt] < 3000]

# Linear regression
f, ax = plt.subplots()
# sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
#                              "axes.labelsize":12})

# scatter plot
# Averages for the +/- 63 age group
y_less_63 = dftt[tt][(dftt[A] <= 63)].dropna()
y_greater_63 = dftt[tt][(dftt[A] > 63)].dropna()

f, ax = plt.subplots()
ax = sns.scatterplot(data=dftt, x=A, y=tt, hue=A63, style=A63, 
                     hue_order=None, 
                     palette='Set1', s=200, legend=True)

ax.get_legend_handles_labels()[0][0]._sizes = [200.]
ax.get_legend_handles_labels()[0][1]._sizes = [200.]

legend = ax.legend(loc='best', fontsize=18, title=A_G)

plt.setp(legend.get_title(), fontsize=18)

# Axis labels
ax.set_xlabel(A_yrs, fontsize=18)
ax.set_ylabel(CohDMGUnit, fontsize=18)


x_plot_less_63 = np.linspace(30, 63, 100)
x_plot_greater_63 = np.linspace(63, 80, 100)

# Plot averages
plt.plot(x_plot_less_63, np.mean(y_less_63)*np.ones(len(x_plot_less_63)), 
         '-.', color='r', linewidth=3) # , label=r'Age $\leq$ 60 AVG')

ax.text(np.mean(x_plot_less_63)*1.1, np.mean(y_less_63) + 50, 
        r'Average', color='r', horizontalalignment='left', 
        fontsize=18, weight='semibold')

plt.plot(x_plot_greater_63, 
         np.mean(y_greater_63)*np.ones(len(x_plot_greater_63)), '-.', 
         color='b', linewidth=3) #, label=r'Age $>$ 60 AVG')

ax.text(np.mean(x_plot_greater_63)*0.9, np.mean(y_greater_63) - 75, 
        r'Average', color='b', horizontalalignment='left', 
        fontsize=18, weight='semibold')

ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# New path
NP = os.path.join(SF, Folder)

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

plt.savefig(os.path.join(NP, 'tt_vs_Age.pdf'), bbox_inches='tight')

plt.close()
# In[Fracture Energy group plots]

FE = 'FE'

Folder = 'FractureEnergy'

# Filter data (Brittany)
dfFE = md[md[FE] < 0.0009]

dfFilt = {FE: dfFE}

for key, val in dfFilt.items():
    
    pivotFE = pd.pivot_table(val, values=key, index=[A60, R], 
                                                 aggfunc=pvtOut)
    
    print('pivotFE')
    print(pivotFE)
    # Add the index groups and convert NaN's to "-"'s
    print(pivotFE.to_latex(index=True, na_rep='-', escape=False, 
                           float_format="{:0.3f}".format))

    # Barplot
    smartPlot(data=val, x=A60, y=key, hue=R, hue_order=[Eq, Po], ci=68, 
              errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
              sigLoc='outside', text_format='star', line_offset=0.015, 
              line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
              legLoc='best', verbose=2, yAxis='log', 
              xlabel=A_G, ylabel=FEUnit, legendTitle=R, 
              figName='Region_BarPlot', folderName=Folder)
    
    # Boxplot
    smartPlot(data=val, x=A60, y=key, hue=R, hue_order=[Eq, Po], plot='boxplot', 
              test='t-test_ind', sigLoc='outside', text_format='star', 
              line_offset=0.015, line_offset_to_box=0.0, line_height=0.015, 
              fontsize='small', legLoc='best', verbose=2, yAxis='log', 
              xlabel=A_G, ylabel=FEUnit, 
              legendTitle=R, figName='Region_BoxPlot', folderName=Folder)
    
    # Boxplot with data
    smartPlot(data=val, x=A60, y=key, hue=R, hue_order=[Eq, Po], plot='boxplot', 
              test='t-test_ind', sigLoc='outside', text_format='star', 
              line_offset=0.015, line_offset_to_box=0.0, line_height=0.015, 
              fontsize='small', legLoc='best', verbose=2, yAxis='log', 
              xlabel=A_G, ylabel=FEUnit, 
              legendTitle=R, 
              figName='Region_BoxPlotWithData', folderName=Folder, 
              dataPoints=True)


# In[Summary Convergence Table Elastic]

# value summary
simList = []

for key, val in ElasticSummary.items():
    d = val[1:] # Subset - skip first row
    d['simTime'] = pd.to_numeric(d['simTime'], downcast="float")
    
    L = len(d.index)
    s = np.sum(d['simTime'])
    avg = s/L
    
    simList.append([L, s, avg])

simDF = pd.DataFrame(simList, columns=['N', 'TotalTime', 'AVGTime'])

print(np.mean(simDF['N']), 
      np.mean(simDF['TotalTime']), 
      np.mean(simDF['AVGTime']))

# In[Summary Convergence Table Cohesive]

# value summary
simList = []

for key, val in CohesiveSummary.items():
    d = val[1:] # Subset - skip first row
    d['simTime'] = pd.to_numeric(d['simTime'], downcast="float")
    
    L = len(d.index)
    s = np.sum(d['simTime'])
    avg = s/L
    
    simList.append([L, s, avg])

simDF = pd.DataFrame(simList, columns=['N', 'TotalTime', 'AVGTime'])

print(np.mean(simDF['N']), 
      np.mean(simDF['TotalTime']), 
      np.mean(simDF['AVGTime']))

