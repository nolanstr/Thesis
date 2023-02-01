# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 23:59:51 2021

@author: Kiffer
"""

import pandas as pd
import numpy as np
import seaborn as sns
from statannot import add_stat_annotation
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
from scipy import stats
import pdb
import os
import glob
import re
from scipy import stats

cwd = os.getcwd()

SF = os.path.join('Results', 'StatisticsFigures')

# Create folder if it doesn't exist
os.makedirs(SF, exist_ok=True)

# In[KDE plot function]

def KDEplot(data=None, x=None, hue=None, hue_order=None, 
            Regions=None, figName=None, legendTitle=None, legendLoc=None, 
            xlabel=None, ylabel=None, bw_adjust=None, alpha=None, 
            initGuess=None, constraints=None, folderName=None, 
            optLegendLoc=None, bounds=None, log_scale=None, cut=None):
    colors = [plt.cm.tab10.colors[i:i + 2] for i in 
              range(0, len(data[R].unique()) * 2, 2)]
    hatches = ['', '//////']
    
    f, ax = plt.subplots(figsize=(9.6, 6))
    sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                                 "axes.labelsize":12})
    handles = []
    for region, palette in zip(Regions, colors):
        
        # Data subset
        dataSubset = data[(data[R] == region)]
        
        # KDE plot
        ax = sns.kdeplot(data=dataSubset, x=x, hue=hue, 
                         hue_order=hue_order, multiple='stack', fill=True, 
                         palette=palette, ax=ax, log_scale=log_scale, 
                         alpha=alpha, bw_adjust=bw_adjust) # , cut=cut
        # pdb.set_trace()
        for h, age, hatch in zip(ax.legend_.legendHandles, hue_order, 
                                 hatches):
            h.set_label(f'{region}, {age}')
            h.set_hatch(hatch)
            handles.append(h)
    
    
    extra = []
    if initGuess != None:
        ax.axvline(x = initGuess, color='black', linestyle='-', 
                   linewidth=1, label=r'Initial Guess')
        extra.append(0)
    
    if bounds != None and bounds == True:
        # Only add bounds if "True"
        if constraints != None:
            ax.axvline(x = constraints[0], color='black', linestyle=':', 
                       linewidth=1, label=r'Lower Bound', ymax=0.4)
            extra.append(1)
            ax.axvline(x = constraints[1], color='black', linestyle='--', 
                       linewidth=1, label=r'Upper Bound', ymax=0.4)
            extra.append(2)
    
    ax.legend_.remove() # remove the automatic legends
    ax.set(xlabel=xlabel, ylabel=ylabel)
    for collection, hatch in zip(ax.collections[::-1],
                                 hatches * len(Regions)):
        collection.set_hatch(hatch)
    
    # Add bounds
    if initGuess != None and optLegendLoc !=None:
        lines = f.gca().get_lines()
        # pdb.set_trace()
        legend2 = ax.legend([lines[i] for i in extra], 
                            [lines[i].get_label() for i in extra], 
                            prop={"size":10}, loc=optLegendLoc, 
                            title='Optimization')
        ax.add_artist(legend2)
    
    legend1 = ax.legend(handles=handles, loc=legendLoc).set_title(legendTitle)
    
    # This doesn't work using the method so the bounds need to be plotted 
    # before the custom legend with handles
    # plt.gca().add_artist(legend1)
    
    if folderName != None:
        # If a new folder name is given, put the files there
        
        # New file path
        NP = os.path.join(SF, folderName)
        
        # Create folder if it doesn't exist
        os.makedirs(NP, exist_ok=True)
        
    else:
        # Put the file in the same folder
        NP = SF
    
    f.savefig(os.path.join(NP, f'{figName}_{x}.pdf'), 
                    bbox_inches='tight')
    
    plt.close(f)
    
    return f, ax


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

# In[Load data]

data = os.path.join(cwd, 'Results', 'OutputFiles', 'ExpSimSummary.txt')
df = pd.read_csv(data, sep = '\t')

df.loc[df[R] == 'Eq.', R] = Eq
df.loc[df[R] == 'Po.', R] = Po

# In[Elastic Modulus distribution]

Folder = 'ElasticModulus'

# Normal distribution plots
f, ax = plt.subplots(figsize=(9.6, 6))
sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
                             "axes.labelsize":12})
ax = sns.kdeplot(data=df, x='EV', fill=True, legend=False, palette='Paired', 
                 cut=0, bw_adjust=0.5)

ax.set(xlabel=ElasticUnit, ylabel=KDEUnit)

# New path
NP = os.path.join(SF, Folder)

# Make folder if it doesn't exist
os.makedirs(NP, exist_ok=True)

plt.savefig(os.path.join(NP, 'Distribution_AllData.pdf'), 
            bbox_inches='tight')

# In[Distributions of cohesive parameters]

alpha = 0.4
bw_adjust = 0.8 # Normal distribution smoothing (smaller is less smooth)
figName = 'Optimization'
F1 = 'ElasticModulus'
F2 = 'CohesiveBehavior'
F3 = 'CohesiveDamage'
F4 = 'FractureEnergy'

# Bandwidth selector
bwE = 1.06*np.std(df['EV'])*np.count_nonzero(df['EV'])**(-1/5)
bwE = 0.9*min(np.std(df['EV']), 
              stats.iqr(df['EV']))*np.count_nonzero(df['EV'])**(-1/5)

bwKnn = 0.9*min(np.std(df['Knn']), 
                stats.iqr(df['Knn']))*np.count_nonzero(df['Knn'])**(-1/5)

# Elastic Modulus
f, ax = KDEplot(data=df, x='EV', hue=A60, hue_order=[Aleq60, Ag60], 
                Regions=[Eq, Po], figName=figName, folderName=F1, 
                legendLoc='best', legendTitle=R, 
                xlabel=ElasticUnit, ylabel=KDEUnit, 
                bw_adjust=bw_adjust, alpha=alpha, 
                initGuess=172, constraints=[50, 2100], 
                optLegendLoc='center right', bounds=False, log_scale=True)

#, cut=False

# Cohesive Behavior
f, ax = KDEplot(data=df, x='Knn', hue=A60, hue_order=[Aleq60, Ag60], 
                Regions=[Eq, Po], figName=figName, folderName=F2, 
                legendLoc='best', legendTitle=R, 
                xlabel=CohBehUnit, ylabel=KDEUnit, 
                bw_adjust=bw_adjust, alpha=alpha, 
                initGuess=2**20.872765304828103, constraints=[2**10, 2**28], 
                optLegendLoc='center right', bounds=False, log_scale=True)

f, ax = KDEplot(data=df, x='Kss', hue=A60, hue_order=[Aleq60, Ag60], 
                Regions=[Eq, Po], figName=figName, folderName=F2, 
                legendLoc='best', legendTitle=R, 
                xlabel=CohBehUnit, ylabel=KDEUnit, 
                bw_adjust=bw_adjust, alpha=alpha, 
                initGuess=2**26.094732037712763, constraints=[2**10, 2**28], 
                optLegendLoc='center right', bounds=False, log_scale=True)

f, ax = KDEplot(data=df, x='Ktt', hue=A60, hue_order=[Aleq60, Ag60], 
                Regions=[Eq, Po], figName=figName, folderName=F2, 
                legendLoc='best', legendTitle=R, 
                xlabel=CohBehUnit, ylabel=KDEUnit, 
                bw_adjust=bw_adjust, alpha=alpha, 
                initGuess=2**26.20110650892766, constraints=[2**10, 2**28], 
                optLegendLoc='center right', bounds=False, log_scale=True)

# Damage Initiation
f, ax = KDEplot(data=df, x='tn', hue=A60, hue_order=[Aleq60, Ag60], 
                Regions=[Eq, Po], figName=figName, folderName=F3, 
                legendLoc='best', legendTitle=R, 
                xlabel=CohDMGUnit, ylabel=KDEUnit, 
                bw_adjust=bw_adjust, alpha=alpha, 
                initGuess=2**9.712181223168551, constraints=[2**3, 2**20], 
                optLegendLoc='center right', bounds=False, log_scale=True)

f, ax = KDEplot(data=df, x='ts', hue=A60, hue_order=[Aleq60, Ag60], 
                Regions=[Eq, Po], figName=figName, folderName=F3, 
                legendLoc='best', legendTitle=R, 
                xlabel=CohDMGUnit, ylabel=KDEUnit, 
                bw_adjust=bw_adjust, alpha=alpha, 
                initGuess=2**9.931687876075074, constraints=[2**3, 2**20], 
                optLegendLoc='center right', bounds=False, log_scale=True)

f, ax = KDEplot(data=df, x='tt', hue=A60, hue_order=[Aleq60, Ag60], 
                Regions=[Eq, Po], figName=figName, folderName=F3, 
                legendLoc='best', legendTitle=R, 
                xlabel=CohDMGUnit, ylabel=KDEUnit, 
                bw_adjust=bw_adjust, alpha=alpha, 
                initGuess=2**9.022372079206395, constraints=[2**3, 2**20], 
                optLegendLoc='center right', bounds=False, log_scale=True)

# Fracture Energy
f, ax = KDEplot(data=df, x='FE', hue=A60, hue_order=[Aleq60, Ag60], 
                Regions=[Eq, Po], figName=figName, folderName=F4, 
                legendLoc='best', legendTitle=R, 
                xlabel=FEUnit, ylabel=KDEUnit, 
                bw_adjust=bw_adjust, alpha=alpha, 
                initGuess=3.738925970000001e-6, constraints=[2**-30, 2**0], 
                optLegendLoc='center right', bounds=False, log_scale=True)

# In[Stack overflow]

# fig, axs = plt.subplots(ncols=3, figsize=(15, 3), sharex=True, sharey=True)
# f, axs = plt.subplots()
# colors = [plt.cm.tab20.colors[i:i + 2] for i in range(0, len(df_CohDmg['Region'].unique()) * 2, 2)]
# hatches = ['', '//']
# for ax, coh_dmg in zip(axs, ['tn', 'ts', 'tt']):
#     handles = []
    # for region, palette in zip([Eq, Po], colors):
    #     sns.kdeplot(data=df_CohDmg[(df_CohDmg['CohDmg'] == coh_dmg) & 
    #                                (df_CohDmg['Region'] == region)],
    #                 x='value', hue='Age60', hue_order=[Aleq60, Ag60], 
    #                 multiple='stack', palette=palette, ax=ax, log_scale=True,)
    #     for h, age, hatch in zip(ax.legend_.legendHandles, [Aleq60, Ag60], hatches):
    #         h.set_label(f'{region}, {age}')
    #         h.set_hatch(hatch)
    #         handles.append(h)
    # ax.legend_.remove() # remove the automatic legends
    # ax.set_title(f'CohDmg={coh_dmg}')
    # for collection, hatch in zip(ax.collections[::-1], hatches * len([Eq, Po])):
    #     collection.set_hatch(hatch)

    # ax.legend(handles=handles, loc='best')
    # plt.tight_layout()
    
    # fig.savefig(os.path.join(SF, f'StackOverflow_{coh_dmg}.pdf'), 
    #                bbox_inches='tight')

# In[Successful KDF plot example]
# colors = [plt.cm.tab20.colors[i:i + 2] for i in 
#               range(0, len(df_CohDmg['Region'].unique()) * 2, 2)]
# hatches = ['', '//']

# for i in ['tn', 'ts', 'tt']:
#     f, ax = plt.subplots(figsize=(8, 5))
#     sns.set_context("paper", rc={"font.size":12, "axes.titlesize":8, 
#                                  "axes.labelsize":12})
#     handles = []
#     for region, palette in zip([Eq, Po], colors):
#         ax = sns.kdeplot(data=df_CohDmg[(df_CohDmg['CohDmg'] == i) & 
#                                         (df_CohDmg['Region'] == region)],
#                          x='value', hue='Age60', hue_order=[Aleq60, Ag60], 
#                          multiple='stack', palette=palette, ax=ax, 
#                          log_scale=True, alpha=0.9, bw_adjust=0.5)
#         for h, age, hatch in zip(ax.legend_.legendHandles, [Aleq60, Ag60], 
#                                  hatches):
#             h.set_label(f'{region}, {age}')
#             h.set_hatch(hatch)
#             handles.append(h)
#     ax.legend_.remove() # remove the automatic legends
#     ax.set(xlabel='Elastic Modulus [Pa]', 
#            ylabel='Kernel Density Estimation')
#     for collection, hatch in zip(ax.collections[::-1],
#                                  hatches * len([Eq, Po])):
#         collection.set_hatch(hatch)
    
#     ax.legend(handles=handles, loc='best')
    
#     f.savefig(os.path.join(SF, f'StackOverflow_{i}.pdf'), 
#                     bbox_inches='tight')