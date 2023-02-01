# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:48:07 2020

@author: Kiffer
"""
import lmfit as lf # lmfit
import numpy as np # numpy
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 9]
import pandas as pd
import os
import sys
import pdb
filePath = os.getcwd() # Location of Python script

dataSets = ['Jami', 'Lee', 'Kashani', 'Polymer', 'Tram']
dataSet = 4
dataSet = dataSets[dataSet]

def PronyR2(y, fit):
    # R squared calculation
    SS_tot = np.sum((y - np.mean(y))**2)
    SS_res = np.sum((y - fit)**2)
    Rsqd = 1 - SS_res/SS_tot
    return Rsqd

# In[Jami Data for example]
if dataSet == 'Jami':
    #############  Jami Creep Shear Test Data Curve Fit######################
    dataName = 'Jami 2014'
    Jami_data = pd.read_csv('Jami/Jami_Shear_Data.txt', sep="\t", header=0)
    Jami_data.columns = ["Time", "NormShearCreep"]
    Jami_data['NormRelaxData'] = 1/Jami_data.NormShearCreep
    
    # Invert and compute the raw data
    Raw_data_0 = 0.0214671 # first shear point in raw data
    Jami_data['CreepData'] = Jami_data.NormShearCreep*Raw_data_0
    Jami_data['RelaxData'] = 1/Jami_data['CreepData']
    
    # Convert data to array
    t = Jami_data.Time
    # data = Jami_data.NormShearCreep
    data = Jami_data.NormRelaxData # Normalized relaxation data
    # data = Jami_data.CreepData
    # data = Jami_data.RelaxData
    
    # Units for plotting
    CreepUnits = r'$\left(\mathrm{Pa}\right)$'
    RelaxUnits = r'$\left(\frac{1}{\mathrm{Pa}}\right)$'
    HorizontalUnits = r'Time (s)'

# In[Lee Digitized Data]

if dataSet == 'Lee':
    # Read data from Lee1992 Viscoelastic material properties 
    # Digitized from http://getdata-graph-digitizer.com/
    dataName = 'Lee 1992'
    Lee_data = pd.read_csv('Lee1992_DigitizedData.txt', sep="\t", header=3)
    Lee_data.columns = ["Time", "CreepDyne"]
    # Conversion is dyne/cm^2 to Pa is multiply by 0.1
    Lee_data["CreepPa"] = Lee_data.CreepDyne*0.1 
    Lee_data["RelaxPa"] = 1/Lee_data.CreepPa
    Lee_data["CreepNormalized"] = Lee_data.CreepPa/Lee_data.CreepPa[0]
    Lee_data["RelaxNormalized"] = Lee_data.RelaxPa/Lee_data.RelaxPa[0]
    
    t = Lee_data.Time
    data = Lee_data.RelaxNormalized
    
    # Units for plotting
    CreepUnits = r'$\left(\mathrm{Pa}\right)$'
    RelaxUnits = r'$\left(\frac{1}{\mathrm{Pa}}\right)$'
    HorizontalUnits = r'Time (s)'

# In[Kashani_2011]
if dataSet == 'Kashani':
    # Porcine eyes
    t = np.linspace(1, 600, 1000)
    
    def modelEqn(t,J1,J2,T1,T2,eta_m):
        """
        Parameters
        ----------
        J1 : TYPE
            DESCRIPTION.
        J2 : TYPE
            DESCRIPTION.
        T1 : TYPE
            DESCRIPTION.
        T2 : TYPE
            DESCRIPTION.
        eta_m : TYPE
            DESCRIPTION.
    
        Returns
        -------
        Compliance
        """
        return J1*(1 - np.exp(-t/T1)) + J2*(1 - np.exp(-t/T2)) + t/eta_m
    
    J1,J2,T1,T2,eta_m = 1.36, 2.64, 1.77, 1.36, 1332.0
    J_0 = modelEqn(1,J1,J2,T1,T2,eta_m) # Initial value
    data = 1/(modelEqn(t,J1,J2,T1,T2,eta_m)/J_0) # invert to define relaxation
    # I'm not sure how this equation works
    
    # Read data from Lee1992 Viscoelastic material properties 
    # Digitized from http://getdata-graph-digitizer.com/
    dataName = 'Kashani 2011'
    Kashani_data = pd.read_csv('KashaniCreepData.csv', sep=",", header=1)
    Kashani_data.columns = ["Time1", "Creep1", 
                            "Time2", "Creep2", 
                            "Time3", "Creep3"]
    # t = Kashani_data.Time1.dropna() # get rid of NaN
    # Compliance = Kashani_data.Creep1.dropna() # Pa # get rid of NaN
    t = Kashani_data.Time2.dropna() # get rid of NaN
    Creep = Kashani_data.Creep2.dropna() # Pa # get rid of NaN
    # t = Kashani_data.Time3.dropna() # get rid of NaN
    # Compliance = Kashani_data.Creep3.dropna() # Pa # get rid of NaN
    
    CreepNorm = Creep/Creep[0]
    NormRelaxation = 1/CreepNorm
    data = NormRelaxation
    
    # Units for plotting
    CreepUnits = r'$\left(\mathrm{Pa}\right)$'
    RelaxUnits = r'$\left(\frac{1}{\mathrm{Pa}}\right)$'
    HorizontalUnits = r'Time (s)'

# In[polymer data]

if dataSet == 'Polymer':
    ###############  Creep Shear Test Data Curve Fit########################
    dataName = 'Polymer 2019'
    subFolder = 'Polymer'
    # df = pd.read_csv(os.path.join(filePath, subFolder, 
    #                               'Dogbone1_Test1.csv'), 
    #                  sep=",", header=5)
    df = pd.read_csv(os.path.join(filePath, subFolder, 
                                  'Dogbone_1_StressRelaxation_2.csv'), 
                     sep=",", header=5)
    # df = pd.read_csv(os.path.join(filePath, subFolder, 
    #                               'Dogbone1_StressRelaxtion_3.csv'), 
    #                  sep=",", header=5)
    df.columns = ["Time", "Extension","Load"]
    
    # Convert dataframe to array
    time = np.asarray(df['Time'].tolist())
    extension = np.asarray(df['Extension'].tolist())
    load = np.asarray(df['Load'].tolist())
    
    # Specimen properties
    width = 12/1000.0
    thickness = 5.3/1000.0
    
    # calculations
    stress = load/(width*thickness)
    strain = extension/100
    
    # Determine where the stress begins to decrease from the max point 
    # in the array
    StressRelax=stress[np.argmax(stress):-1];
    TimeRelax=time[np.argmax(stress):-1]-time[np.argmax(stress)]
    
    data = StressRelax#/StressRelax[0] # Normalized stress relaxation
    t = TimeRelax
    
    # Units for plotting
    CreepUnits = r'$\left(\mathrm{Pa}\right)$'
    RelaxUnits = r'$\left(\frac{1}{\mathrm{Pa}}\right)$'
    HorizontalUnits = r'Displacement (x)'

# In[Tram]

if dataSet == 'Tram':
    dataName = 'Tram'
    ExcelPath = 'Tram_2018_Creep_Data.xlsx'
    
    df = pd.read_excel(os.path.join(filePath, ExcelPath), sheet_name=None)
    
    path = 'Tram'
    
    # Folder for general figures to be stored
    TramFigures = os.path.join(path, 'Figures')
    if not os.path.exists(TramFigures):
        os.makedirs(TramFigures)
    
    for i,j in enumerate(df.keys()):
        if j == 'HU 0764 OS 1 Pa': # 9
        # if j == 'HU2018-0074 OD 1 Pa': # 5
        # if j =='HU2018-0125 OS 1 Pa': # 4
        # if j == 'HU2018-0125 OD 1 Pa': # 3
            
            # Specific file name (adds the iteration number for organization)
            specificName = str(i) + '_' + j
            
            # Make directory for each data trace and associated images
            specificPath = os.path.join(path, '{}_'.format(i) + j)
            if not os.path.exists(specificPath):
                os.makedirs(specificPath)
            
            sheeti = df[j] #.dropna() # Eliminate rows with NA
            
            time = sheeti.iloc[1:-1,0].reset_index(drop=True)
            creep = sheeti.iloc[1:-1,1].reset_index(drop=True)
            
            time_constantStress = 6
            # Shift values past region of ramp stress
            creep = creep[time >= time_constantStress].reset_index(drop=True)
            # Shift values past region of ramp stress
            time = time[time >= time_constantStress].reset_index(drop=True)
            
            # Convert pandas series to numpy arrays
            timeArray = time.to_numpy(dtype='float')
            creepArray = creep.to_numpy(dtype='float')
            
            # Get rid of nan values from the data trace
            timeArrayRemoveNans = timeArray[np.logical_not(np.isnan(creepArray))]
            creepArrayRemoveNans = creepArray[np.logical_not(np.isnan(creepArray))]
            
            # Start time at 0
            timeArrayRemoveNans = timeArrayRemoveNans - timeArrayRemoveNans[0]
    
    t = timeArrayRemoveNans
    creepData = creepArrayRemoveNans
    creepNorm = creepData/creepData[0]
    data = 1/creepNorm

# In[Lmfit]

def residual(pars, t, data=None):
    """
    Parameters
    ----------
    pars : g_k and Tau_k for Prony N'th order terms
            The final parameter is the sum of the terms that needs to be less
            than 1 for realistic thermodynamic properties
    t : time array being passed in
    data : Normalized relaxation data (1/creep compliance) to be passed
            through to compare to model data.  The default is None.

    Returns
    -------
    If no data is supplied, the return is the new model for plotting the final
    curve
    
    If data is supplied it will calculate the error between the actual
    and known data
    """
    
    # Extract g_k and tau_k from the pars class variable
    g_k = []
    tau_k = []
    for key, value in pars.items():
        if key.find('g') >= 0:
            g_k.append(value.value)
        elif key.find('T') >= 0:
            tau_k.append(value.value)
    
    if NormalizedData is True:
        model = 1 # Normalized so this begins at 1
    else:
        G0 = pars['G0'].value # Instantaneous modulus
        model = G0
    for i in range(len(g_k)):
        model -= g_k[i]*(1 - np.exp(-t/tau_k[i])) # Loop over prony terms
    
    if data is None:
        return model
    return model - data

def dresidual(pars, t, data=None):
    """
    Derivative of the function to return the jacobian for faster optimization
    Parameters
    ----------
    pars : g_k and Tau_k for Prony N'th order terms
            The final parameter is the sum of the terms that needs to be less
            than 1 for realistic thermodynamic properties
    t : time array being passed in
    data : Normalized relaxation data (1/creep compliance) to be passed
            through to compare to model data.  The default is None.
    Returns
    -------
    The jacobian (partial derivatives with respect to unknown variables)
    """
    # Extract g_k and tau_k from the pars class variable
    g_k = []
    tau_k = []
    for key, value in pars.items():
        if key.find('g') >= 0:
            g_k.append(value.value)
        elif key.find('T') >= 0:
            tau_k.append(value.value)
    jac = []
    if NormalizedData is not True:
        jac.append(np.ones(len(t))) # derivative of G(t) with respect to G0
    for i in range(len(g_k)):
        jac.append(-1 + np.exp(-t/tau_k[i]))
        jac.append(g_k[i]*t*np.exp(-t/tau_k[i])/tau_k[i]**2)
    return np.asarray(jac)


def PronyN(N, jac=None):
    """
    Parameters
    ----------
    N : Number of parameters in the Prony Series fit
    
    sumG ensures that the values for the individual springs divided by the G0
    value sum to a value less than 1.
    
    Returns
    -------
    out : Model output
    t_fit : Model time output
    fit : Model fit output
    """
    # Specify paramters bounds with a for loop for N terms
    fit_params = lf.Parameters() # intialize the class for parameters
    if NormalizedData is True:
        # Used when normalized data
        fit_params.add('G0', value = 1, vary=False) 
    else:
        # Instantaneous shear modulus
        fit_params.add('G0', value = 1, min=0, max=G_0_UpperLimit) 
    sumG = ''
    for i in range(N):
        if i == N-1:
            # append g_k values for the constraint eqn
            sumG = sumG + 'g_{}/G0'.format(i + 1) 
        else:
            # append g_k values for the constraint eqn
            sumG = sumG + 'g_{}/G0 + '.format(i + 1) 
        
        # If Normalized data the bounds of the values are [0,1], 
        # otherwise [0,infinity]
        if NormalizedData is True:
            # Used when normalized
            fit_params.add('g_{}'.format(i + 1), value=0.1/N, 
                           min=0.0, max=1.0)
            fit_params.add('Tau_{}'.format(i + 1), value=1, min=0.0)
        else:
            # Used when not normalized (1/N)
            fit_params.add('g_{}'.format(i + 1), value=1, min=0.0)
            # Polymer 0.00001*G_0_UpperLimit
            fit_params.add('Tau_{}'.format(i + 1), value=1, min=0.0)
    
    # comment this out if you want to relax the requirement for the 
    # sum of coefficients
    if Constraint is True:
        # Constraint eqn
        fit_params.add('sumG', min=0, max=1, expr=sumG, vary=True)
    
    # Set up minimization class to be able to pass derivative in (Jacobian)
    minClass = lf.Minimizer(residual, fit_params, fcn_args=(t,), 
                            fcn_kws={'data': data})
    if jac is None:
        # No jacobian
        out = minClass.leastsq()
    else:
        # Yes jacobian
        out = minClass.leastsq(Dfun=dresidual, col_deriv=1)
    # t_fit = np.linspace(0,max(t), 1000)
    fit = residual(out.params, t) # t_fit
    
    lf.report_fit(out) # modelpars=p_true,  show_correl=True
    print('\n\n\n')
    return out, t, fit # t_fit

# In[Prony series curve fit]
""" Information for running the curve-fit algorithm """
pronyTerms = [1,2,3,4] # number of prony series terms to be plotted [List]
# Upper limit on the instantaneous shear modulus
G_0_UpperLimit = 1000 # 1000000 
NormalizedData = True # Normalized data
# enforce the constraint where the sum of G_k's can't be more than 1
Constraint = True 
nu = 0.49
jac = True # Jacobian (if None, then don't include.  If True, do include)
if jac is None:
    print('No Jacobian')
elif jac is True:
    print('Jacobian')
elif jac is False:
    print('jac needs to be "None" or "True"')
    sys.exit()

print('Upper limit for instantaneous shear modulus is', G_0_UpperLimit)

p = {} # empty dictionary
tfit = {} # empty dictionary
f = {} # empty dictionary

# Loop over the number of prony terms to calculate the curve fit paramters
for i in pronyTerms:
    A, B, C = PronyN(i, jac)
    p['p{}'.format(i)] = A
    tfit['tfit{}'.format(i)] = B
    f['f{}'.format(i)] = C

# In[Plot data]
E_0 = 1#1840
v_0 = 0.49
G_0 = E_0/(2*(1+v_0))
G_0 = 49.3075445
def Prony2(t,a,b,c,d):
    return G_0*(1 - a*(1 - np.exp(-t/b)) - c*(1 - np.exp(-t/d)))
ABQtime = np.linspace(0, max(t), 300)
ABQfit_Norm = Prony2(0, 0.70134, 2.96389e-2, 0.19334, 0.47088)
ABQfit = Prony2(ABQtime, 0.70134, 2.96389e-2, 0.19334, 0.47088)

# Plot Relaxation
# plt.plot(ABQtime, ABQfit, 'b-', label='ABAQUS')
plt.plot(t, data, '.', label=dataName + ' Data')
for i in pronyTerms:
    plt.plot(tfit['tfit{}'.format(i)], f['f{}'.format(i)], 
             label='LMFIT {} Prony terms'.format(i) if jac is None else 
             'LMFIT {} Prony terms with Jacobian'.format(i))
# plt.ylim(0.01,0.02)
plt.xlabel('Time (s)',fontsize=18)
plt.ylabel(r'Relaxation Modulus',fontsize=18) # Normalized
plt.title('Viscoelastic Response',fontsize=20)
plt.legend(loc = 'best', fontsize=14)
plt.grid(True, which='both')
plt.savefig("Figures/1LmFitRelax.pdf" if jac is None else 
            "Figures/1LmFitRelaxJac.pdf", bbox_inches='tight')
plt.show()


# Plot Compliance
# plt.plot(ABQtime, 1/ABQfit, 'b-', label='ABAQUS')

color_map = plt.cm.tab10

plt.plot(t, 1/data, ':o', label=dataName + ' Data', linewidth=2, 
         markersize=5, color=color_map.colors[0])

# Plot only the 4 term fit instead of all fits
for i in pronyTerms[-1:]:
    rsqrd_i = PronyR2(1/data, 1/f[f'f{i}'])
    plt.plot(tfit[f'tfit{i}'], 1/f[f'f{i}'], 
             label=f'LMFIT {i} Prony terms' if jac is None else 
             f'LMFIT {i} Prony terms with Jacobian, $r^2={rsqrd_i:.5}$', 
             linewidth=2, color=color_map.colors[1])

# plt.ylim(50,100)
plt.xlabel('Time (s)',fontsize=18)
plt.ylabel(r'Normalized Creep Compliance',fontsize=18)
# plt.title('Viscoelastic Response',fontsize=20)
plt.legend(loc = 'best', fontsize=14)
plt.grid(True, which='both', alpha=0.5)
plt.savefig("Figures/2LmFitCompliance.pdf" if jac is None else 
            "Figures/2LmFitComplianceJac.pdf", bbox_inches='tight')
plt.show()

# LogLog (To show different regions (Elastic, Retardation, Viscous))
plt.loglog(t, data, '.', label=dataName + ' Data')
for i in pronyTerms:
    plt.loglog(tfit['tfit{}'.format(i)], f['f{}'.format(i)], 
               label='LMFIT {} Prony terms'.format(i) if jac is None else 
               'LMFIT {} Prony terms with Jacobian'.format(i))
plt.xlabel('Time (s)',fontsize=18)
plt.ylabel(r'Relaxation Modulus',fontsize=18)
plt.title('Viscoelastic Response',fontsize=20)
plt.legend(loc = 'best', fontsize=14)
plt.grid(True, which='both')
plt.savefig("Figures/3LmFitRelaxLogLog.pdf" if jac is None else 
            "Figures/3LmFitRelaxJacLogLog.pdf", bbox_inches='tight')
plt.show()

# Plot Compliance
plt.loglog(t, 1/data, '.', label=dataName + ' Data')
for i in pronyTerms:
    plt.loglog(tfit['tfit{}'.format(i)], 1/f['f{}'.format(i)], 
               label='LMFIT {} Prony terms'.format(i) if jac is None else 
               'LMFIT {} Prony terms with Jacobian'.format(i))
plt.xlabel('Time (s)',fontsize=18)
plt.ylabel(r'Creep Compliance',fontsize=18)
plt.title('Viscoelastic Response',fontsize=20)
plt.legend(loc = 'best', fontsize=14)
plt.grid(True, which='both')
plt.savefig("Figures/4LmFitComplianceLogLog.pdf" if jac is None else 
            "Figures/4mFitComplianceJacLogLog.pdf", bbox_inches='tight')
plt.show()

# In[Normalized Plots]
# if NormalizedData == True:
#     # Plot Relaxation
#     data0 = data[0]
#     plt.plot(t, data/data0, '.', label=dataName + ' Data')
#     for i in pronyTerms:
#         ti = tfit['tfit{}'.format(i)] # Time
#         fi = f['f{}'.format(i)] # Curve fit data
#         fi0 = fi[0] # Normalization by the first data point data0# 
        # plt.plot(ti, fi/fi0, 
        #          label='LMFIT {} Prony terms'.format(i) if jac is None else 
        #           'LMFIT {} Prony terms with Jacobian'.format(i))
#     # plt.ylim(0.01,0.02)
#     plt.xlabel('Time (s)',fontsize=18)
#     plt.ylabel(r'Normalized Relaxation Modulus',fontsize=18) # Normalized
#     plt.title('Viscoelastic Response',fontsize=20)
#     plt.legend(loc = 'best', fontsize=14)
#     plt.grid(True, which='both')
    # plt.savefig("Figures/5NormLmFitRelax.pdf" if jac is None else 
    #             "Figures/5NormLmFitRelaxJac.pdf", bbox_inches='tight')
#     plt.show()
    
#     # Plot Compliance
#     plt.plot(t, 1/(data/data0), '.', label=dataName + ' Data')
#     for i in pronyTerms:
#         ti = tfit['tfit{}'.format(i)] # Time
#         fi = f['f{}'.format(i)] # Curve fit data
#         fi0 = fi[0] # Normalization by the first data point data0# 
        # plt.plot(ti, 1/(fi/fi0), 
        #          label='LMFIT {} Prony terms'.format(i) if jac is None else 
        #          'LMFIT {} Prony terms with Jacobian'.format(i))
#     # plt.ylim(50,100)
#     plt.xlabel('Time (s)',fontsize=18)
#     plt.ylabel(r'Normalized Creep Compliance',fontsize=18)
#     plt.title('Viscoelastic Response',fontsize=20)
#     plt.legend(loc = 'best', fontsize=14)
#     plt.grid(True, which='both')
    # plt.savefig("Figures/6NormLmFitCompliance.pdf" if jac is None else 
    #             "Figures/6NormLmFitComplianceJac.pdf", bbox_inches='tight')
#     plt.show()
    
#     # LogLog (To show different regions (Elastic, Retardation, Viscous))
#     plt.loglog(t, data/data0, '.', label=dataName + ' Data')
#     for i in pronyTerms:
#         ti = tfit['tfit{}'.format(i)] # Time
#         fi = f['f{}'.format(i)] # Curve fit data
#         fi0 = fi[0] # Normalization by the first data point data0# 
        # plt.loglog(ti, fi, 
        #            label='LMFIT {} Prony terms'.format(i) if jac is None else 
        #            LMFIT {} Prony terms with Jacobian'.format(i))
#     plt.xlabel('Time (s)',fontsize=18)
#     plt.ylabel(r'Normalized Relaxation Modulus',fontsize=18)
#     plt.title('Viscoelastic Response',fontsize=20)
#     plt.legend(loc = 'best', fontsize=14)
#     plt.grid(True, which='both')
    # plt.savefig("Figures/7NormLmFitRelaxLogLog.pdf" if jac is None else 
    #             "Figures/7NormLmFitRelaxJacLogLog.pdf", 
    #             bbox_inches='tight')
#     plt.show()
    
#     # Plot Compliance
#     plt.loglog(t, 1/(data/data0), '.', label=dataName + ' Data')
#     for i in pronyTerms:
#         ti = tfit['tfit{}'.format(i)] # Time
#         fi = f['f{}'.format(i)] # Curve fit data
#         fi0 = fi[0] # Normalization by the first data point data0# 
        # plt.loglog(ti, 1/(fi/fi0), 
        #            label='LMFIT {} Prony terms'.format(i) if jac is None else 
        #            'LMFIT {} Prony terms with Jacobian'.format(i))
#     plt.xlabel('Time (s)',fontsize=18)
#     plt.ylabel(r'Normalized Creep Compliance',fontsize=18)
#     plt.title('Viscoelastic Response',fontsize=20)
#     plt.legend(loc = 'best', fontsize=14)
#     plt.grid(True, which='both')
    # plt.savefig("Figures/8NormLmFitComplianceLogLog.pdf" if jac is None else 
    #             "Figures/8NormLmFitComplianceJacLogLog.pdf", 
    #             bbox_inches='tight')
#     plt.show()


# In[Extract Data]
# Extract data from LMFIT report
optParams = {} # empty dictionary
# Loop over pronyTerms to extract the g_k and tau_k values for each N'th 
# order fit
for i in pronyTerms:
    g_k = [] # Shear modulus per Prony element
    tau_k = [] # Time constant
    G0_k = [] # Instantaneous shear modulus
    for key, value in p['p{}'.format(i)].params.items():
        if key.find('g') >= 0:
            g_k.append(value.value)
        elif key.find('T') >= 0:
            tau_k.append(value.value)
        elif key.find('G0') >= 0:
            G0_k.append(value.value)
    optParams['P{}'.format(i)] = g_k
    optParams['T{}'.format(i)] = tau_k
    optParams['G0{}'.format(i)] = G0_k

# Write vitreous creep compliance data to a txt file for abaqus importing
file1 = open('Vitreous_Prony_Constants_LMFIT.txt' if jac is None else 
             'Vitreous_Prony_Constants_LMFIT_Jac.txt' ,"w") 
str1 = ("Equation is in the form:  G(t) = G_0*(1 - " + 
        "SUM_i^N(g_k^P*(1 - exp(-t/tau_k))))")
str2 = '\t'.join(["Prony_#", "g_k^P", "k_i", "tau_k"])
str3 = 'Data set = ' + dataName
str4 = '' if jac is None else 'Jacobian was used to converge'
str5 = '' if Constraint is False else ("enforce the constraint where the " + 
                                       "sum of G_k/G_0's can't be more than 1")
str6 = '' if NormalizedData is True else ("Upper limit for G_0 is" + 
                                          " {}".format(G_0_UpperLimit))
file1.write('\n'.join([str1, str2 ,str3, str4, str5, str6]) + '\n')

# Loop over pronyTerm results to write to a txt file
for i in pronyTerms:
    g_k = optParams['P{}'.format(i)] # shear modulus of prony term
    tau_k = optParams['T{}'.format(i)] # Time constant
    G_0 = optParams['G0{}'.format(i)][0] # Instantaneous shear modulus
    file1.write('\n' + 79*'=')
    file1.write('\n' + 22*' ' + 'Prony series order ' + str(len(g_k)))
    file1.write('\nCalculated instantaneous shear modulus (G_0) is: ' + 
                str(G_0))
    file1.write('\nKnown nu = {}'.format(nu))
    file1.write("\nCalculated instantaneous Young's modulus (E_0) is: " + 
                str(G_0*2*(1 + nu)) + '\n')
    file1.write('\nNormalized coefficients\n')
    for m in range(len(g_k)):
        file1.write(''.join(['(' + str(g_k[m]),', 0.0, ', 
                             str(tau_k[m]) + '),']) + '\n')
    print(sum(g_k),"g_k")
    file1.write("\n")
    # file1.write('Normalized coefficients (g_k/G_0)\n')
    # for m in range(len(g_k)):
        # file1.write(''.join(['(' + str(g_k[m]/G_0),', 0.0, ', 
        #                      str(tau_k[m]) + '),']) + '\n')
    # print(sum(g_k),"g_k")
    # file1.write("\n")
file1.close()