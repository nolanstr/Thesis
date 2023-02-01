# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:48:15 2020

@author: Kiffer Creveling
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns
from statannot import add_stat_annotation
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
plt.rcParams['figure.figsize'] = [16, 10]
from scipy import stats
import pdb

# In[Functions]

# fcn for plotting
def yfit(x):
    return slope*x + intercept

# In[Read values from Database]
""" Read from the database """

df = pd.read_csv('JMP_Data.csv') # Data from JMP
df = pd.read_excel('Human Data Paper 2 TEM only (Updated Jul 10 2020).xlsx', 
                   engine='openpyxl')
df = pd.read_excel('Human Data Paper 2 TEM only (Updated April 17 2021).xlsx', 
                   engine='openpyxl')

""" Simplification of code """
SF = 'StatisticsFigures' # Figure directory
TMD = 'TEM Mean Density'
TMA = 'TEM Mean Angle'
TAA = 'TEM Angle ABS'
ILM = 'ILM Thickness (nm)'
FL = 'Fiber Length (um)'
MPF = 'Maximum peel force (mN)'
SSPF = 'Steady-state peel force (mN)'
mpf_mN = 'Max peel force (mN)'
R = 'Region'
Eq = 'Equator'
Po = 'Posterior'
AG = 'AgeGroup'
A60 = 'Age60'
Aleq60 = r'Age $\leq$ 60'
Ag60 = 'Age $>$ 60'
A = 'Age'
MN = 'Max [N]'
MmN = 'Max [mN]'
SSN = 'SS [N]'
SSmN = 'SS [mN]'

# Plot attributes (labels, etc)
A_yrs = 'Age (yr.)'
A_G = 'Age Group (yr.)'
DensityUnit = (r'Collagen Fibril Density $\left(\frac{\mathrm{\#~of~fibrils}}{\mathrm{ILM~length~(nm)}}\right)$')
FibrilLengthUnit = r'Collagen Fibril length ($\mu$m)'
OrientationUnit = r'Collagen Fibril Angle Relative to the ILM $(^{\circ})$'

# convert from N to mN
df[mpf_mN] = df[MN]*1000
df[SSmN] = df[SSN]*1000

# Exclude the cells that have duplicates or have been exculded due to 
# video analysis
df = df[df['Excluded'] != 'yes']

# In[Create AgeGroup bins]
bins = [30, 40, 50, 60, 70, 80, 90]
labels = ['30-39', '40-49', '50-59', '60-69','70-79', '80-89']
# Create binned AgeGroups

df[AG] = pd.cut(df[A], bins, labels=labels, right=False)

bins = [0, 60, 90]
labels = [Aleq60, Ag60]
# Create binned AgeGroups
df[A60] = pd.cut(df[A], bins, labels=labels, right=True)

# In[Pivot Table]
# Simplify pivot table output

pvtOut = {'count', np.median, np.mean, np.std} # pivot table outputs

# In[Plots]

standardError = 68 # Used for confidence intervals

sns.set_theme(context='paper', style='darkgrid', palette="Paired", 
              font_scale=2)
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
                 'axes.spines.top': False}
# White background with ticks and black border lines, Turns grid off
ax = sns.set_style(rc=custom_style)

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
              fontsize=None, legLoc=None, verbose=None, xlabel=None, 
              ylabel=None, legendTitle=None, figName=None, folderName=None, 
              dataPoints=None):
    
    # barplot
    f, ax = plt.subplots()
    
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
    
    #Stats results and significant differences (SR)
    SR = add_stat_annotation(ax, plot=plot, data=data, x=x, y=y, hue=hue, 
                             hue_order=hue_order, box_pairs=box_pairs, 
                             test=test, loc=sigLoc, text_format=text_format, 
                             verbose=verbose, comparisons_correction=None, 
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
            l = plt.legend(handles[0:2], labels[0:2], title=legendTitle)
    
    else:
        if hue != None:
            ax.legend(loc=legLoc).set_title(legendTitle)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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

# Special spacing

def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """
    
    # iterating through Axes instances
    for ax in g.axes:
        
        # iterating through axes artists:
        for c in ax.get_children():
            
            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin + xmax)
                xhalf = 0.5*(xmax - xmin)
                
                # setting new width of box
                xmin_new = xmid - fac*xhalf
                xmax_new = xmid + fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new
                
                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])

# In[TEM mean density by age +/- 60 and region]

""" TEM mean density by age +/- 60 and region """

pivotTEM_MeanDensityAgeGroup60 = pd.pivot_table(df, values=TMD, 
                                                index=[A60, R], 
                                                aggfunc=pvtOut)

print('pivotTEM_MeanDensityAgeGroup60')
print(pivotTEM_MeanDensityAgeGroup60)
# Add the index groups and convert NaN's to "-"'s
print(pivotTEM_MeanDensityAgeGroup60.to_latex(index=True, na_rep='-', 
                                              escape=False, 
                                              float_format="{:0.3f}".format))

Folder = 'Density_Age60Region'

# Barplot
smartPlot(data=df, x=A60, y=TMD, hue=R, hue_order=[Eq, Po], ci='sd', 
          errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
          sigLoc='outside', text_format='star', line_offset=0.0, 
          line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
          legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=DensityUnit, legendTitle=R, 
          figName='BarPlot', folderName=Folder)

# Boxplot
smartPlot(data=df, x=A60, y=TMD, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', text_format='star', sigLoc='outside', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=DensityUnit, 
          legendTitle=R, figName='BoxPlot', folderName=Folder)

# Boxplot with data
smartPlot(data=df, x=A60, y=TMD, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=DensityUnit, 
          legendTitle=R, figName='BoxPlotWithData', folderName=Folder, 
          dataPoints=True)


# In[TEM mean density grouped by region]

""" TEM mean density """

pivotTEM_MeanDensityRegion = pd.pivot_table(df, values=TMD, index=[R], 
                                            aggfunc=pvtOut)

print('pivotTEM_MeanDensityRegion')
print(pivotTEM_MeanDensityRegion)
# Add the index groups and convert NaN's to "-"'s
print(pivotTEM_MeanDensityRegion.to_latex(index=True, na_rep='-', 
                                          escape=False, 
                                          float_format="{:0.3f}".format))

Folder = 'Density_Region'

# Barplot
smartPlot(data=df, x=R, y=TMD, hue=None, hue_order=[Eq, Po], ci='sd', 
          errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
          sigLoc='outside', text_format='star', line_offset=0.0, 
          line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
          legLoc='best', verbose=2, 
          xlabel=R, ylabel=DensityUnit, legendTitle=R, 
          figName='BarPlot', folderName=Folder)

# Boxplot
smartPlot(data=df, x=R, y=TMD, hue=None, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=R, ylabel=DensityUnit, 
          legendTitle=R, figName='BoxPlot', folderName=Folder)

# Boxplot with data
smartPlot(data=df, x=R, y=TMD, hue=None, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=R, ylabel=DensityUnit, 
          legendTitle=R, figName='BoxPlotWithData', folderName=Folder, 
          dataPoints=True)

# matched_pairs student's t-test
dfTMD = df[df[TMD].notna()]

dfMP = dfTMD[dfTMD.duplicated(['MatchingID'], keep=False)]
f, p = stats.ttest_rel(dfMP[TMD][dfMP[R] == Eq], 
                       dfMP[TMD][dfMP[R] == Po])

print(f, p, "Matched Pairs Student's t-test")

f, p = stats.ttest_ind(dfTMD[TMD][dfTMD[R] == Eq], 
                       dfTMD[TMD][dfTMD[R] == Po])

print(f, p, "Student's t-test")

# In[TEM mean density grouped by age group decade and region]

pivotTEM_MeanDensity = pd.pivot_table(df, values=TMD, index=[R, AG], 
                                      aggfunc=pvtOut)

print('pivotTEM_MeanDensity')
print(pivotTEM_MeanDensity)
# Add the index groups and convert NaN's to "-"'s
print(pivotTEM_MeanDensity.to_latex(index=True, na_rep='-', escape=False, 
                                    float_format="{:0.3f}".format))

Folder = 'Density_AgeDecadeRegion'

# Barplot
smartPlot(data=df, x=AG, y=TMD, hue=R, hue_order=[Eq, Po], ci='sd', 
          errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
          sigLoc='outside', text_format='star', line_offset=0.0, 
          line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
          legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=DensityUnit, legendTitle=R, 
          figName='BarPlot', folderName=Folder)

# Boxplot
smartPlot(data=df, x=AG, y=TMD, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=DensityUnit, 
          legendTitle=R, figName='BoxPlot', folderName=Folder)

# Boxplot with data
smartPlot(data=df, x=AG, y=TMD, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=DensityUnit, 
          legendTitle=R, 
          figName='BoxPlotWithData', folderName=Folder, 
          dataPoints=True)

# In[ILM thickness vs region age +/- 60]

""" TEM ILM thickness vs region age +/- 60 """

pivotTEM_ILM_ThicknessAge60 = pd.pivot_table(df, values=ILM, index=[A60, R], 
                                             aggfunc=pvtOut)

print('pivotTEM_ILM_ThicknessAge60')
print(pivotTEM_ILM_ThicknessAge60)
# Add the index groups and convert NaN's to "-"'s
print(pivotTEM_ILM_ThicknessAge60.to_latex(index=True, na_rep='-', 
                                           escape=False, 
                                           float_format="{:0.3f}".format))

Folder = 'ILM_Age60Region'

# Barplot
smartPlot(data=df, x=A60, y=ILM, hue=R, hue_order=[Eq, Po], ci='sd', 
          errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
          sigLoc='outside', text_format='star', line_offset=0.0, 
          line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
          legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=ILM, legendTitle=R, 
          figName='BarPlot', folderName=Folder)

# Boxplot
smartPlot(data=df, x=A60, y=ILM, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=ILM, 
          legendTitle=R, figName='BoxPlot', folderName=Folder)

# Boxplot with data
smartPlot(data=df, x=A60, y=ILM, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=ILM, 
          legendTitle=R, figName='BoxPlotWithData', folderName=Folder, 
          dataPoints=True)


# In[ILM thickness vs region age group]

""" ILM thickness vs region and age group """

pivotTEM_ILM_Thickness = pd.pivot_table(df, values=ILM, index=[AG, R], 
                                        aggfunc=pvtOut)

print('pivotTEM_ILM_Thickness')
print(pivotTEM_ILM_Thickness)
# Add the index groups and convert NaN's to "-"'s
print(pivotTEM_ILM_Thickness.to_latex(index=True, na_rep='-', 
                                      escape=False, 
                                      float_format="{:0.3f}".format))

Folder = 'ILM_Region'

# Barplot
smartPlot(data=df, x=AG, y=ILM, hue=R, hue_order=[Eq, Po], ci=68, 
          errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
          sigLoc='outside', text_format='star', line_offset=0.0, 
          line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
          legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=ILM, legendTitle=R, 
          figName='BarPlot', folderName=Folder)

# Boxplot
smartPlot(data=df, x=AG, y=ILM, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=ILM, 
          legendTitle=R, figName='BoxPlot', folderName=Folder)

# Boxplot with data
smartPlot(data=df, x=AG, y=ILM, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=ILM, 
          legendTitle=R, 
          figName='BoxPlotWithData', folderName=Folder, 
          dataPoints=True)


# In[ILM fiber length vs region age group decade]

""" TEM ILM fiber length """

pivotTEM_FiberLength = pd.pivot_table(df, values=FL, index=[AG, R], 
                                      aggfunc=pvtOut)

print('pivotTEM_FiberLength')
print(pivotTEM_FiberLength)
# Add the index groups and convert NaN's to "-"'s
print(pivotTEM_FiberLength.to_latex(index=True, na_rep='-', 
                                    escape=False, 
                                    float_format="{:0.3f}".format))

Folder = 'FibrilLength_AgeDecadeRegion'

# Barplot
smartPlot(data=df, x=AG, y=FL, hue=R, hue_order=[Eq, Po], ci=68, 
          errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
          sigLoc='outside', text_format='star', line_offset=0.0, 
          line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
          legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=FibrilLengthUnit, legendTitle=R, 
          figName='BarPlot', folderName=Folder)

# Boxplot
smartPlot(data=df, x=AG, y=FL, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=FibrilLengthUnit, 
          legendTitle=R, figName='BoxPlot', folderName=Folder)

# Boxplot with data
smartPlot(data=df, x=AG, y=FL, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=FibrilLengthUnit, 
          legendTitle=R, 
          figName='BoxPlotWithData', folderName=Folder, 
          dataPoints=True)


# In[ILM fiber length vs region age group +/- 60]

""" TEM ILM fiber length """

pivotTEM_FiberLengthAge60 = pd.pivot_table(df, values=FL, index=[A60, R], 
                                           aggfunc=pvtOut)

print('pivotTEM_FiberLengthAge60')
print(pivotTEM_FiberLengthAge60)
# Add the index groups and convert NaN's to "-"'s
print(pivotTEM_FiberLengthAge60.to_latex(index=True, na_rep='-', 
                                         escape=False, 
                                         float_format="{:0.3f}".format))

Folder = 'FibrilLength_Age60Region'

# Barplot
smartPlot(data=df, x=A60, y=FL, hue=R, hue_order=[Eq, Po], ci=68, 
          errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
          sigLoc='outside', text_format='star', line_offset=0.0, 
          line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
          legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=FibrilLengthUnit, legendTitle=R, 
          figName='BarPlot', folderName=Folder)

# Boxplot
smartPlot(data=df, x=A60, y=FL, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=FibrilLengthUnit, 
          legendTitle=R, figName='BoxPlot', folderName=Folder)

# Boxplot with data
smartPlot(data=df, x=A60, y=FL, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=FibrilLengthUnit, 
          legendTitle=R, 
          figName='BoxPlotWithData', folderName=Folder, 
          dataPoints=True)

# In[TEM Absolute Angle by age +/- 60 and region]

""" TEM Absolute Angle """

pivotTEM_MeanAngleABSAgeGroup60 = pd.pivot_table(df, values=TAA, 
                                                 index=[A60, R], 
                                                 aggfunc=pvtOut)

print('pivotTEM_MeanAngleABSAgeGroup60')
print(pivotTEM_MeanAngleABSAgeGroup60)
# Add the index groups and convert NaN's to "-"'s
print(pivotTEM_MeanAngleABSAgeGroup60.to_latex(index=True, na_rep='-', 
                                               escape=False, 
                                               float_format="{:0.3f}".format))

Folder = 'ABSAngle_Age60Region'

# Barplot
smartPlot(data=df, x=A60, y=TAA, hue=R, hue_order=[Eq, Po], ci=68, 
          errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
          sigLoc='outside', text_format='star', line_offset=0.0, 
          line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
          legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=OrientationUnit, legendTitle=R, 
          figName='BarPlot', folderName=Folder)

# Boxplot
smartPlot(data=df, x=A60, y=TAA, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=OrientationUnit, 
          legendTitle=R, figName='BoxPlot', folderName=Folder)

# Boxplot with data
smartPlot(data=df, x=A60, y=TAA, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=OrientationUnit, 
          legendTitle=R, 
          figName='BoxPlotWithData', folderName=Folder, 
          dataPoints=True)


# In[TEM angle]

pivotTEM_MeanAngle = pd.pivot_table(df, values=TMA, index=[R, AG], 
                                    aggfunc=pvtOut)

print('pivotTEM_MeanAngle')
print(pivotTEM_MeanAngle)
# Add the index groups and convert NaN's to "-"'s
print(pivotTEM_MeanAngle.to_latex(index=True, na_rep='-', 
                                  escape=False, 
                                  float_format="{:0.3f}".format))

OrientationUnitNoAbs = r'ILM angle $(^{\circ})$'
Folder = 'Angle_AgeRegion'

# Barplot
smartPlot(data=df, x=AG, y=TMA, hue=R, hue_order=[Eq, Po], ci=68, 
          errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
          sigLoc='outside', text_format='star', line_offset=0.0, 
          line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
          legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=OrientationUnitNoAbs, legendTitle=R, 
          figName='BarPlot', folderName=Folder)

# Boxplot
smartPlot(data=df, x=AG, y=TMA, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=OrientationUnitNoAbs, 
          legendTitle=R, figName='BoxPlot', folderName=Folder)

# Boxplot with data
smartPlot(data=df, x=AG, y=TMA, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=OrientationUnitNoAbs, 
          legendTitle=R, 
          figName='BoxPlotWithData', folderName=Folder, 
          dataPoints=True)

# In[TEM ABS angle by age decade group and region]

pivotTEM_MeanAngleABS = pd.pivot_table(df, values=TAA, index=[R, AG], 
                                       aggfunc=pvtOut)
print('pivotTEM_MeanAngleABS')
print(pivotTEM_MeanAngleABS)
# Add the index groups and convert NaN's to "-"'s
print(pivotTEM_MeanAngleABS.to_latex(index=True, na_rep='-', 
                                     escape=False, 
                                     float_format="{:0.3f}".format))

Folder = 'ABSAngle_AgeDecadeRegion'

# Barplot
smartPlot(data=df, x=AG, y=TAA, hue=R, hue_order=[Eq, Po], ci=68, 
          errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
          sigLoc='outside', text_format='star', line_offset=0.0, 
          line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
          legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=OrientationUnit, legendTitle=R, 
          figName='BarPlot', folderName=Folder)

# Boxplot
smartPlot(data=df, x=AG, y=TAA, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=OrientationUnit, 
          legendTitle=R, figName='BoxPlot', folderName=Folder)

# Boxplot with data
smartPlot(data=df, x=AG, y=TAA, hue=R, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=A_G, ylabel=OrientationUnit, 
          legendTitle=R, 
          figName='BoxPlotWithData', folderName=Folder, 
          dataPoints=True)

# In[TEM absolute angle by region]

pivotTEM_MeanAngleABSRegion = pd.pivot_table(df, values=TAA, index=[R], 
                                             aggfunc=pvtOut)

print('pivotTEM_MeanAngleABSRegion')
print(pivotTEM_MeanAngleABSRegion)
# Add the index groups and convert NaN's to "-"'s
print(pivotTEM_MeanAngleABSRegion.to_latex(index=True, na_rep='-', 
                                           escape=False, 
                                           float_format="{:0.3f}".format))

Folder = 'ABSAngle_Region'

# Barplot
smartPlot(data=df, x=R, y=TAA, hue=None, hue_order=[Eq, Po], ci=68, 
          errcolor='black', capsize=.2, plot='barplot', test='t-test_ind', 
          sigLoc='outside', text_format='star', line_offset=0.0, 
          line_offset_to_box=0.0, line_height=0.015, fontsize='small', 
          legLoc='best', verbose=2, 
          xlabel=R, ylabel=OrientationUnit, legendTitle=R, 
          figName='BarPlot', folderName=Folder)

# Boxplot
smartPlot(data=df, x=R, y=TAA, hue=None, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=R, ylabel=OrientationUnit, 
          legendTitle=R, figName='BoxPlot', folderName=Folder)

# Boxplot with data
smartPlot(data=df, x=R, y=TAA, hue=None, hue_order=[Eq, Po], plot='boxplot', 
          test='t-test_ind', sigLoc='outside', text_format='star', 
          line_offset=0.0, line_offset_to_box=0.0, line_height=0.015, 
          fontsize='small', legLoc='best', verbose=2, 
          xlabel=R, ylabel=OrientationUnit, 
          legendTitle=R, 
          figName='BoxPlotWithData', folderName=Folder, 
          dataPoints=True)

# In[ILM thickness vs age regression]

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
# dict(Equator="r", Posterior="b") , 'color':'black', 'color':'blue'
ax = sns.lmplot(x=A, y=ILM, hue=R, markers=["o", "x"], data=df, 
                legend_out=False, fit_reg=True, height=5, aspect=1.6, 
                palette="Set1", truncate=False, ci=95, line_kws={'lw':0})
ax.set(ylabel=ILM, xlabel=A_yrs)

# Remove all NaN's from the data for regressions

# remove nans from ILM thickness
df_no_Nan = df.dropna(subset=[ILM])

# linear regressions for fitting
x = df_no_Nan[A][df_no_Nan[R] == Eq]
y = df_no_Nan[ILM][df_no_Nan[R] == Eq]

x_plot = np.linspace(min(x), max(x), 100)

slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)
plt.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1, label='line')
plt.text(80, yfit(80) + 20, r'$r={:.4f}$'.format(r_value1), color='r', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

# linear regressions for fitting
x = df_no_Nan[A][df_no_Nan[R] == Po]
y = df_no_Nan[ILM][df_no_Nan[R] == Po]

x_plot = np.linspace(min(x), max(x), 100)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)
plt.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1, label='line')
plt.text(75, yfit(75) + 20, r'$r={:.4f}$'.format(r_value2), color='b', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

# Axis limits
ax.set(ylim=(0, None))
ax.set(xlim=(None, None))

# New path
NP = os.path.join(SF, 'ILM_vs_Age')

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

ax.savefig(os.path.join(NP, 'Regression.pdf'), bbox_inches='tight')
plt.close()

# In[Max peel force vs ILM thickness]

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
ax = sns.lmplot(x=ILM, y=mpf_mN, hue=R, markers=["o", "x"], data=df, 
                legend_out=False, fit_reg=True, height=5, aspect=1.6, 
                palette="Set1", truncate=True, ci=95, line_kws={'lw':0})
ax.set(xlabel=ILM, ylabel=MPF)

# Remove all NaN's from the data for regressions
# remove nans from ILM thickness & Max
df_no_Nan = df.dropna(subset=[ILM, mpf_mN])

# linear regressions for fitting
x = df_no_Nan[ILM][df_no_Nan[R] == Eq]
# Convert to N
y = df_no_Nan[mpf_mN][df_no_Nan[R] == Eq]

x_plot = np.linspace(min(x), max(x), 100)

slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)
plt.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1, label='line')
plt.text(500, yfit(500) + 4, r'$r={:.4f}$'.format(r_value1), color='r', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'ILM thickness and Max Force in the Equator\n', 
      'P={:.4f}'.format(p_value), 'r={:.4f}'.format(r_value1))

# linear regressions for fitting
x = df_no_Nan[ILM][df_no_Nan[R] == Po]
y = df_no_Nan[mpf_mN][df_no_Nan[R] == Po]


x_plot = np.linspace(min(x), max(x), 100)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)
plt.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1, label='line')
plt.text(1500, yfit(1500) + 1, r'$r={:.4f}$'.format(r_value2), color='b', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between ' + 
      'ILM thickness and Max Force in the Posterior\n', 
      f'P={p_value:.4f}', f'r={r_value2:.4f}')

# Axis limits
ax.set(ylim=(0, 18))
ax.set(xlim=(0, max(x)*1.1))

# New path
NP = os.path.join(SF, 'ILM_vs_MaxPeel')

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

ax.savefig(os.path.join(NP, 'Regression.pdf'), bbox_inches='tight')
plt.close()

# In[Max peel force vs ILM thickness by age group]

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
ax = sns.lmplot(x=ILM, y=mpf_mN, hue=A60, markers=["o", "x"], data=df, 
                legend_out=False, fit_reg=True, height=5, aspect=1.6, 
                palette="Set1", truncate=True, ci=95, line_kws={'lw':0})
ax.set(xlabel=ILM, ylabel=MPF)

# Remove all NaN's from the data for regressions
# remove nans from ILM thickness & Max
df_no_Nan = df.dropna(subset=[ILM, mpf_mN])

# linear regressions for fitting
x = df_no_Nan[ILM][df_no_Nan[A60] == Aleq60]
y = df_no_Nan[mpf_mN][df_no_Nan[A60] == Aleq60] # MmN

x_plot = np.linspace(min(x), max(x), 100)

# linear regression
slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)

# Linear regression line
plt.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1)
plt.text(1250, yfit(1250) + 0.75, r'$r={:.4f}$'.format(r_value1), color='r', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

# linear regressions for fitting
x = df_no_Nan[ILM][df_no_Nan[A60] == Ag60]
y = df_no_Nan[mpf_mN][df_no_Nan[A60] == Ag60] # MmN

x_plot = np.linspace(min(x), max(x), 100)
# linear regression
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)

plt.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1) # linear regression
plt.text(1000, yfit(1000) + 1, r'$r={:.4f}$'.format(r_value2), color='b', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

# Legend
plt.legend(loc='best').set_title(A_G) # legend

# axis limits
ax.set(ylim=(0, 18))
ax.set(xlim=(0, 2200))

# New path
NP = os.path.join(SF, 'ILM_vs_MaxPeel_Age60')

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

ax.savefig(os.path.join(NP, 'Regression.pdf'), bbox_inches='tight')
plt.close()


# In[Max peel force vs ILM thickness in the Equator]

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
ax = sns.lmplot(x=ILM, y=mpf_mN, hue=A60, markers=["o", "x"], 
                data=df[df[R] == Eq], legend_out=False, fit_reg=True, height=5, 
                aspect=1.6, palette="Set1", truncate=False, ci=95, 
                line_kws={'lw':0})
ax.set(xlabel=ILM, ylabel=MPF)

# Remove all NaN's from the data for regressions
# remove nans from ILM thickness & Max
df_no_Nan = df.dropna(subset=[ILM, mpf_mN])

# linear regressions for fitting
x = df_no_Nan[ILM][(df_no_Nan[A60] == Aleq60) & (df[R] == Eq)]
y = df_no_Nan[mpf_mN][(df_no_Nan[A60] == Aleq60) & (df[R] == Eq)] # MmN

x_plot = np.linspace(min(x), max(x), 100)

# linear regression
slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)

# Linear regression line
plt.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1)
plt.text(500, yfit(500) + 1, r'$r={:.4f}$'.format(r_value1), color='r', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

# linear regressions for fitting
x = df_no_Nan[ILM][(df_no_Nan[A60] == Ag60) & (df[R] == Eq)]
y = df_no_Nan[mpf_mN][(df_no_Nan[A60] == Ag60) & (df[R] == Eq)] # MmN

x_plot = np.linspace(min(x), max(x), 100)
# linear regression
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)

plt.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1) # linear regression
plt.text(500, yfit(500) - 1, r'$r={:.4f}$'.format(r_value2), color='b', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

# Legend
plt.legend(loc='best').set_title("Equator Age group (yr.)") # legend

# axis limits
ax.set(ylim=(0, 20))
# ax.set(xlim=(0, None))

# New path
NP = os.path.join(SF, 'ILM_vs_MaxPeel_Age60_Equator')

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

ax.savefig(os.path.join(NP, 'Regression.pdf'), bbox_inches='tight')
plt.close()

# In[Steady state peel force vs ILM density]

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
ax = sns.lmplot(x=TMD, y=SSmN, hue=R, markers=["o", "x"], data=df, 
                legend_out=False, fit_reg=True, height=5, aspect=1.6, 
                palette="Set1", truncate=True, ci=95, line_kws={'lw':0})
ax.set(xlabel=DensityUnit, ylabel=SSPF)

# Remove all NaN's from the data for regressions
# remove nans from ILM thickness & Max
df_no_Nan = df.dropna(subset=[TMD, SSmN])
# figure out why zero's aren't being eliminiated

# linear regressions for fitting
x = df_no_Nan[TMD][df_no_Nan[R] == Eq]
y = df_no_Nan[SSmN][df_no_Nan[R] == Eq]

x_plot = np.linspace(min(x), max(x), 100)

slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)
plt.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1, label='line')
plt.text(85, yfit(85) + 0.2, r'$r={:.4f}$'.format(r_value1), color='r', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between Steady-state and Equator\n', 
      f'P={p_value:.4f}', f'r={r_value1:.4f}')

# linear regressions for fitting
x = df_no_Nan[TMD][df_no_Nan[R] == Po]
y = df_no_Nan[SSmN][df_no_Nan[R] == Po]

x_plot = np.linspace(min(x), max(x), 100)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)
plt.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1, label='line')
plt.text(70, yfit(70) + 0.3, r'$r={:.4f}$'.format(r_value2), color='b', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Values for correlation between Steady-state and Posterior\n', 
      f'P={p_value:.4f}', f'r={r_value2:.4f}')

# axis limits
ax.set(ylim=(0, None))
ax.set(xlim=(0, max(df_no_Nan[TMD])*1.05))

# New path
NP = os.path.join(SF, 'Density_vs_SteadyStatePeel_Region')

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

ax.savefig(os.path.join(NP, 'Regression.pdf'), bbox_inches='tight')
plt.close()


# In[Maximum peel force vs ILM density]

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
ax = sns.lmplot(x=TMD, y=mpf_mN, hue=R, markers=["o", "x"], data=df, 
                legend_out=False, fit_reg=True, height=5, aspect=1.6, 
                palette="Set1", truncate=False, ci=95, line_kws={'lw':0})
ax.set(xlabel=DensityUnit, ylabel='Maximum peel force (mN)')

# Remove all NaN's from the data for regressions
# remove nans from ILM thickness & Max
df_no_Nan = df.dropna(subset=[TMD, mpf_mN])
# figure out why zero's aren't being eliminiated

# linear regressions for fitting
x = df_no_Nan[TMD][df_no_Nan[R] == Eq]
y = df_no_Nan[mpf_mN][df_no_Nan[R] == Eq]

x_plot = np.linspace(min(x), max(x), 100)

slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)
plt.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1, label='line')
plt.text(85, yfit(85) + 0.1, r'$r={:.4f}$'.format(r_value1), color='r', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

# linear regressions for fitting
x = df_no_Nan[TMD][df_no_Nan[R] == Po]
y = df_no_Nan[mpf_mN][df_no_Nan[R] == Po]

x_plot = np.linspace(min(x), max(x), 100)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)
plt.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1, label='line')
plt.text(70, yfit(70) + 0.1, r'$r={:.4f}$'.format(r_value2), color='b', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

# axis limits
ax.set(ylim=(0, None))
# ax.set(xlim=(0, None))

# New path
NP = os.path.join(SF, 'Density_vs_MaxPeel_Region')

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

ax.savefig(os.path.join(NP, 'Regression.pdf'), bbox_inches='tight')
plt.close()

# In[Collagen fibril density vs age correlation (regression)]

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
# dict(Equator="r", Posterior="b") , 'color':'black', 'color':'blue'
ax = sns.lmplot(x=A, y=TMD, hue=R, markers=["o", "x"], data=df, 
                legend_out=False, fit_reg=True,height=5, aspect=1.6, 
                palette="Set1", truncate=False, ci=95, line_kws={'lw':0})
ax.set(ylabel=DensityUnit, xlabel=A_yrs)

# Remove all NaN's from the data for regressions

# remove nans from ILM thickness
df_no_Nan = df.dropna(subset=[TMD])

# linear regressions for fitting
x = df_no_Nan[A][df_no_Nan[R] == Eq]
y = df_no_Nan[TMD][df_no_Nan[R] == Eq]

x_plot = np.linspace(min(x), max(x), 100)

slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)
plt.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1, label='line')
plt.text(80, yfit(80) + 5, r'$r={:.4f}$'.format(r_value1), color='r', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

# linear regressions for fitting
x = df_no_Nan[A][df_no_Nan[R] == Po]
y = df_no_Nan[TMD][df_no_Nan[R] == Po]

x_plot = np.linspace(min(x), max(x), 100)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)
plt.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1, label='line')
plt.text(75, yfit(75) + 5, r'$r={:.4f}$'.format(r_value2), color='b', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

# Axis limits
ax.set(ylim=(0, None))
ax.set(xlim=(None, None))

# New path
NP = os.path.join(SF, 'Density_vs_Age')

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

ax.savefig(os.path.join(NP, 'Regression.pdf'), bbox_inches='tight')
plt.close()


# In[Collagen fibril Orientation vs age correlation (regression)]

# Linear regression
f, ax = plt.subplots()
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
# dict(Equator="r", Posterior="b") , 'color':'black', 'color':'blue'
ax = sns.lmplot(x=A, y=TAA, hue=R, markers=["o", "x"], data=df, 
                legend_out=False, fit_reg=True,height=5, aspect=1.6, 
                palette="Set1", truncate=False, ci=95, line_kws={'lw':0})
ax.set(ylabel=OrientationUnit, xlabel=A_yrs)

# Remove all NaN's from the data for regressions

# remove nans from ILM thickness
df_no_Nan = df.dropna(subset=[TAA])

# linear regressions for fitting
x = df_no_Nan[A][df_no_Nan[R] == Eq]
y = df_no_Nan[TAA][df_no_Nan[R] == Eq]

x_plot = np.linspace(min(x), max(x), 100)

slope, intercept, r_value1, p_value, std_err = stats.linregress(x, y)
plt.plot(x_plot, yfit(x_plot), '-', color='r', linewidth=1, label='line')
plt.text(80, yfit(80) + 2, r'$r={:.4f}$'.format(r_value1), color='r', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Collagen fibril Equator orientation\n', 
      f'P={p_value:.4f}', f'r={r_value1:.4f}')

# linear regressions for fitting
x = df_no_Nan[A][df_no_Nan[R] == Po]
y = df_no_Nan[TAA][df_no_Nan[R] == Po]

x_plot = np.linspace(min(x), max(x), 100)
slope, intercept, r_value2, p_value, std_err = stats.linregress(x, y)
plt.plot(x_plot, yfit(x_plot), '-', color='b', linewidth=1, label='line')
plt.text(75, yfit(75) + 2, r'$r={:.4f}$'.format(r_value2), color='b', 
         horizontalalignment='left', fontsize=8, weight='semibold') # r value

print('Collagen fibril Posterior orientation\n', 
      f'P={p_value:.4f}', f'r={r_value2:.4f}')

# Axis limits
ax.set(ylim=(0, None))
ax.set(xlim=(None, None))

# New path
NP = os.path.join(SF, 'Angle_vs_Age')

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

ax.savefig(os.path.join(NP, 'Regression.pdf'), bbox_inches='tight')
plt.close()

# In[Collagen fibril orientation distributions]

# remove nans from ILM thickness
df_no_Nan = df.dropna(subset=[TAA])

# Normal distribution plots
f, ax = plt.subplots(figsize=(9.6, 6))
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})

ax = sns.kdeplot(data=df_no_Nan, x=TAA, hue=R, hue_order=[Eq, Po], fill=True, 
                 legend=False, palette='Paired', multiple='layer', 
                 cut=0, bw_adjust=0.7, alpha=0.3)

ax.set(xlabel=OrientationUnit, ylabel='Kernel Density Estimation')

# Legend
plt.legend(labels=[Eq, Po], loc='best').set_title(R)

# Axis limits
# ax.set(ylim=(0, None))
# ax.set(xlim=(0, 90))

# New path
NP = os.path.join(SF, 'Angle')

# Create folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

plt.savefig(os.path.join(NP, 'Distribution.pdf'), bbox_inches='tight')