# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:20:21 2020

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

dataSets = ['Jami', 'Lee', 'Kashani', 'Polymer']
dataSet = 1
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
    # data = Jami_data.NormRelaxData # Normalized relaxation data
    data = Jami_data.CreepData
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
    Lee_data.columns = ["Time", "Compliance"]
    
    t = Lee_data.Time
    
    # Conversion is dyne/cm^2 to Pa is multiply by 0.1
    data = Lee_data.Compliance*0.1 
    
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
    Kashani_data.columns = ["Time1", "Compliance1", 
                            "Time2", "Compliance2", 
                            "Time3", "Compliance3"]
    t = Kashani_data.Time1.dropna() # get rid of NaN
    Compliance = Kashani_data.Compliance1.dropna() # Pa # get rid of NaN
    t = Kashani_data.Time2.dropna() # get rid of NaN
    Compliance = Kashani_data.Compliance2.dropna() # Pa # get rid of NaN
    # t = Kashani_data.Time3.dropna() # get rid of NaN
    # Compliance = Kashani_data.Compliance3.dropna() # Pa # get rid of NaN
    
    data = Compliance
    
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
    
    # Determine where the stress begins to decrease from the max point in 
    # the array
    StressRelax=stress[np.argmax(stress):-1];
    TimeRelax=time[np.argmax(stress):-1]-time[np.argmax(stress)]
    
    data = 1/StressRelax
    t = TimeRelax
    
    # Units for plotting
    CreepUnits = r'$\left(\mathrm{Pa}\right)$'
    RelaxUnits = r'$\left(\frac{1}{\mathrm{Pa}}\right)$'
    HorizontalUnits = r'Displacement (x)'

# In[Lmfit]

def residual(pars, t, data=None):
    """
    Parameters
    ----------
    pars : Ee and E1 and tau2 for Visco terms for the Standard Linear Solid 
            Model from Lin2020
    t : time array being passed in
    data : Data to be passed
            through to compare to model data.  The default is None.

    Returns
    -------
    If no data is supplied, the return is the new model for plotting the final
    curve
    
    If data is supplied it will calculate the error between the actual
    and known data
    """
    
    Ee = pars['Ee'].value # Instantaneous modulus
    E1 = pars['E1'].value # Instantaneous modulus
    tau2 = pars['tau2'].value # Time constant
    
    # Standard Linear Solid Model (SLSM)
    model = 1/Ee*(1 - E1/(E1 + Ee)*np.exp(-t/tau2))
    
    if data is None:
        return model
    return model - data

def dresidual(pars, t, data=None):
    """
    Derivative of the function to return the jacobian for faster optimization
    Parameters
    ----------
    pars : Ee and E1 and tau2 for Visco terms for the Standard Linear Solid 
            Model from Lin2020
    t : time array being passed in
    data : Normalized relaxation data (1/creep compliance) to be passed
            through to compare to model data.  The default is None.
    Returns
    -------
    The jacobian (partial derivatives with respect to unknown variables)
    """
    
    Ee = pars['Ee'].value # Instantaneous modulus
    E1 = pars['E1'].value # Instantaneous modulus
    tau2 = pars['tau2'].value # Time constant
    
    jac = []
    
    dCdEe = ((E1*Ee + (E1 + Ee)*(E1 - (E1 + Ee)*np.exp(t/tau2)))*
             np.exp(-t/tau2)/(Ee**2*(E1 + Ee)**2))
    dCdE1 = -np.exp(-t/tau2)/(E1 + Ee)**2
    dCdtau2 = -E1*t*np.exp(-t/tau2)/(Ee*tau2**2*(E1 + Ee))
    
    jac.append(dCdEe)
    jac.append(dCdE1)
    jac.append(dCdtau2)
    return np.asarray(jac)

def SLSM(jac=None):
    """
    Parameters
    ----------
    Add parameters to be fit using the SLS model
    
    Returns
    -------
    out : Model output
    t : Model time output
    fit : Model fit output
    """
    
    # Specify parameters
    fit_params = lf.Parameters() # intialize the class for parameters
    fit_params.add('Ee', value = 1, min=0) # Instantaneous shear modulus
    fit_params.add('E1', value = 1, min=0) # Total change in modulus
    fit_params.add('tau2', value = 1) # Time constant
    
    # Set up minimization class to be able to pass derivative in (Jacobian)
    minClass = lf.Minimizer(residual, fit_params, fcn_args=(t,), 
                            fcn_kws={'data': data})
    if jac is None:
        # No jacobian
        out = minClass.leastsq()
    else:
        # Yes jacobian
        out = minClass.leastsq(Dfun=dresidual, col_deriv=1)
    fit = residual(out.params, t) # run the model to fit the data
    
    lf.report_fit(out) # modelpars=p_true,  show_correl=True
    print('\n\n\n')
    return out, t, fit

# In[Prony series curve fit]
""" Information for running the curve-fit algorithm """
nu = 0.49
jac = True # Jacobian (if None, then don't include.  If True, do include)

if jac is None:
    print('No Jacobian')
elif jac is True:
    print('Jacobian')
elif jac is False:
    print('jac needs to be "None" or "True"')
    sys.exit()

p = {} # empty dictionary
tfit = {} # empty dictionary
f = {} # empty dictionary

# run the curve fit
p['p'], tfit['tfit'], f['f'] = SLSM(jac)

# In[Plot data]

color_map = plt.cm.tab10

# Plot Relaxation
plt.plot(t, data, 'o', label=dataName + ' Data', linewidth=2, markersize=5, 
         color=color_map.colors[0])

rsqrd = PronyR2(data, f['f'])

plt.plot(tfit['tfit'], f['f'], 
         label='LMFIT 2-Term Standard Linear Solid Model' if jac is None else 
         'LMFIT 2-Term Standard Linear Solid Model with Jacobian, ' +  
         f'$r^2={rsqrd:.5}$', linewidth=2, color=color_map.colors[1])
plt.xlabel(HorizontalUnits, fontsize=18)
plt.ylabel(r'Creep Response ' + CreepUnits, fontsize=18)
plt.legend(loc = 'best', fontsize=14)
plt.grid(True, which='both', alpha=0.5)
plt.savefig("Figures/1LmFitSLSMCreep.pdf" if jac is None else 
            "Figures/1LmFitSLSMCreepJac.pdf", bbox_inches='tight')
plt.show()

# Plot Compliance
plt.plot(t, 1/data, '.', label=dataName + ' Data')
plt.plot(tfit['tfit'], 1/f['f'], 
         label='LMFIT Standard Linear Solid Model' if jac is None else 
         'LMFIT Standard Linear Solid Model with Jacobian')
plt.xlabel(HorizontalUnits, fontsize=18)
plt.ylabel(r'Relaxation Response ' + RelaxUnits, fontsize=18)
plt.legend(loc = 'best', fontsize=14)
plt.grid(True, which='both')
plt.savefig("Figures/2LmFitSLSMRelax.pdf" if jac is None else 
            "Figures/2LmFitSLSMRelaxJac.pdf", bbox_inches='tight')
plt.show()

# LogLog (To show different regions (Elastic, Retardation, Viscous))
plt.loglog(t, data, '.', label=dataName + ' Data')
plt.loglog(tfit['tfit'], f['f'], 
           label='LMFIT Standard Linear Solid Model' if jac is None else 
           'LMFIT Standard Linear Solid Model with Jacobian')
plt.xlabel(HorizontalUnits, fontsize=18)
plt.ylabel(r'Creep Response ' + CreepUnits, fontsize=18)
plt.legend(loc = 'best', fontsize=14)
plt.grid(True, which='both')
plt.savefig("Figures/3LmFitSLSMCreepLogLog.pdf" if jac is None else 
            "Figures/3LmFitSLSMCreepJacLogLog.pdf", bbox_inches='tight')
plt.show()

# Plot Compliance
plt.loglog(t, 1/data, '.', label=dataName + ' Data')
plt.plot(tfit['tfit'], 1/f['f'], 
         label='LMFIT Standard Linear Solid Model' if jac is None else 
         'LMFIT Standard Linear Solid Model with Jacobian')
plt.xlabel(HorizontalUnits, fontsize=18)
plt.ylabel(r'Relaxation Response ' + RelaxUnits, fontsize=18)
plt.legend(loc = 'best', fontsize=14)
plt.grid(True, which='both')
plt.savefig("Figures/4LmFitSLSMRelaxLogLog.pdf" if jac is None else 
            "Figures/4mFitSLSMRelaxJacLogLog.pdf", bbox_inches='tight')
plt.show()

# In[Extract Data]
# Extract data from LMFIT report

Ee = p['p'].params['Ee'].value
E1 = p['p'].params['E1'].value
Tau2 = p['p'].params['tau2'].value

# Write vitreous creep compliance data to a txt file for abaqus importing
file1 = open('Vitreous_SLSM_Constants_LMFIT.txt' if jac is None else 
             'Vitreous_SLSM_Constants_LMFIT_Jac.txt' ,"w") 
str1 = ("Equation is in the form:  C(t) = 1/Go*(1 - G1/(G1" + 
        " + Go)*np.exp(-t/tau2)) # Standard Linear Solid Model (SLSM)")
str2 = ('Standard Linear Solid Model paper Lin2020 Figure 3, ' + 
        'equation 2 used in optimization')
str3 = 'Data set = ' + dataName
str4 = '' if jac is None else 'Jacobian was used to converge'
file1.write('\n'.join([str1, str2, str3, str4]) + '\n')

Eo = E1 + Ee # Instantaneous modulus

nu = 0.49
G_to_E_conversion = 2*(1+nu)

# Write to a txt file
file1.write('\n' + 79*'=')
file1.write('\nCalculated long term modulus (Goo) is: ' + str(Ee))
file1.write("\nCalculated modulus (G1) is: " + str(E1))
file1.write("\nCalculated instantaneous modulus (Go) is: " + str(Eo))
file1.write("\nCalculated time constant (Tau2) is: " + str(Tau2))
file1.write('\n' + 79*'.')
file1.write('\n E = G*2*(1+nu), where nu = {}'.format(nu))
file1.write('\nCalculated long term modulus (Eoo) is: ' + 
            str(Ee*G_to_E_conversion))
file1.write("\nCalculated modulus (E1) is: " + str(E1*G_to_E_conversion))
file1.write("\nCalculated instantaneous modulus (Eo) is: " + 
            str(Eo*G_to_E_conversion))
file1.write("\n")
file1.close()



