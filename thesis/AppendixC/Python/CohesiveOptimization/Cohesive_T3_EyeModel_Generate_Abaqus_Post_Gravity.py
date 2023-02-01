# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:39:42 2021

@author: Kiffer Creveling
"""

"""  abaqus cae -noGUI abaqusMacros.py """

# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__

import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import numpy as np
import os
import sys


# # location of the folder
# # specific folder path where this file is located # os.getcwd()
# pythonScriptPath = os.path.abspath("file")
# abqWD, pythonFiles = os.path.split(pythonScriptPath) # split file path

# # StepFile = 'Adult Human Eye holder Assembly.STEP'
# # StepFile = 'Adult Human Eye holder Assembly 2 Step.STEP'
# # Constrained
# StepFile = 'Adult Human Eye holder Assembly Constrained Bottom.STEP'

# SolidWorksDir = 'SolidWorksStepFiles' # Folder name

# # Combine folder directory
# SolidWorksStepFile = os.path.join(SolidWorksDir, StepFile)

modelName = 'PostODB'
modelDescription = 'Test test test model post gravity'

abqModelPG = mdb.Model(name=modelName, 
                       description=modelDescription, 
                       modelType=STANDARD_EXPLICIT, 
                       restartJob='', 
                       restartStep='Gravity_Step')

tieInterface = False

odbFile = 'F:/Abaqus Working Directory/CohesiveEyeModel/test.odb'
odbName = 'test'

def PostGravity(odbFile):
    ''' Post gravity '''
    
    session.openOdb(odbFile)
    odb = session.odbs[odbFile]
    p = abqModelPG.PartFromOdb(name='E', 
                               instance='E-1', 
                               odb=odb, 
                               shape=DEFORMED, 
                               step=0, 
                               frame=50)
    p = abqModelPG.parts['E']
    
    p = abqModelPG.PartFromOdb(name='R', 
                               instance='R-1', 
                               odb=odb, 
                               shape=DEFORMED, 
                               step=0, 
                               frame=50)
    p = abqModelPG.parts['R']
    
    p = abqModelPG.PartFromOdb(name='V', 
                               instance='V-1', 
                               odb=odb, 
                               shape=DEFORMED, 
                               step=0, 
                               frame=50)
    p = abqModelPG.parts['V']
    
    odb.close()


def Retina_Mat_Prop(RetinaProp):
    retina_E = RetinaProp # Passed in young's modulus
    Retina_Description = """
Actually used the value from Chen 2014
E = 11.12 KPa

------------------------------------------------
Density (kg/m^3)
1100 --------> Esposito_2013

"""
    abqModelPG.Material(name='Retina', description=Retina_Description)
    abqModelPG.materials['Retina'].Density(table=((1100.0, ), ))
    abqModelPG.materials['Retina'].Elastic(table=((retina_E, 0.49), ))
    
    # # Assign the section to the part
    abqModelPG.HomogeneousSolidSection(name='Retina_Section', 
                                       material='Retina', 
                                       thickness=None)

def Vitreous_Mat_Prop(vitreousProp):
    vitreous_E = vitreousProp # Passed in young's modulus
    Vitreous_Description = """
------------------------------------------------
Density (kg/m^3)
950   ------------> Esposito_2013

------------------------------------------------

# Tram 2018 Viscoelasticity data
# 4 Term Prony (Tram Data # 5 HU2018-0074 OD 1 Pa)
(0.1486397420159951, 0.0, 331.4796231072498),
(0.12469207412616717, 0.0, 3.388868494747128),
(0.29059507092540404, 0.0, 15.59692349525066),
(0.1591569334281, 0.0, 69.85134248442381)
"""
    abqModelPG.Material(name='Vitreous', description=Vitreous_Description)
    abqModelPG.materials['Vitreous'].Density(table=((950.0, ), ))
    ''' Using Lin2020 Paper to relate SLSM curve fit parameters to physical 
    values.  Prony 4 Term (Long term) initial guess 172.77874855377468 
    optimization of E'''
    abqModelPG.materials['Vitreous'].Elastic(moduli=LONG_TERM, 
                                           table=((vitreous_E, 0.49), ))
    # Prony 4 Term calculated from normalized data
    abqModelPG.materials['Vitreous'].Viscoelastic(
            domain=TIME, time=PRONY, table=(
            # Tram Data # 5
            (0.1486397420159951, 0.0, 331.4796231072498),
            (0.12469207412616717, 0.0, 3.388868494747128),
            (0.29059507092540404, 0.0, 15.59692349525066),
            (0.1591569334281, 0.0, 69.85134248442381)))
    
    # Assign the section to the part
    abqModelPG.HomogeneousSolidSection(name='Vitreous_Section', 
                                       material='Vitreous', 
                                       thickness=None)


def E_Features():
    ''' Create the reference point for the eye holder from the orphan mesh '''
    p = abqModelPG.parts['E']
    p.ReferencePoint(point=(0.0, 0.0, 0.0))
    n = p.nodes
    nodes = n.getSequenceFromMask(mask=('[#0:39 #40000 ]', ), )
    p.Set(nodes=nodes, name='E-REFPT_')



def R_Features():
    p = abqModelPG.parts['R']
    
    # Assign section
    region = p.sets['R_SET']
    p.SectionAssignment(region=region, 
                        sectionName='Vitreous_Section', 
                        offset=0.0, 
                        offsetType=MIDDLE_SURFACE, 
                        offsetField='', 
                        thicknessAssignment=FROM_SECTION)



def V_Features():
    
    p = abqModelPG.parts['V']
    
    # Assign section
    region = p.sets['V_SET']
    p.SectionAssignment(region=region, 
                        sectionName='Vitreous_Section', 
                        offset=0.0, 
                        offsetType=MIDDLE_SURFACE, 
                        offsetField='', 
                        thicknessAssignment=FROM_SECTION)



def Assembly_ODB():
    ''' Create the assembly from the odb parts '''
    a1 = abqModelPG.rootAssembly
    a1.DatumCsysByDefault(CARTESIAN)
    p = abqModelPG.parts['E']
    a1.Instance(name='E-1', part=p, dependent=ON)
    p = abqModelPG.parts['R']
    a1.Instance(name='R-1', part=p, dependent=ON)
    p = abqModelPG.parts['V']
    a1.Instance(name='V-1', part=p, dependent=ON)




def Step_Post_Gravity(time, scaleFactor, MSTI, stepName, descrip):
    ''' Define the step post gravity '''
    abqModelPG.ExplicitDynamicsStep(name=stepName, 
                                  previous='Initial', 
                                  description=descrip, 
                                  timePeriod=time, 
                                  massScaling=((SEMI_AUTOMATIC, 
                                                MODEL, 
                                                AT_BEGINNING, 
                                                scaleFactor, 
                                                MSTI, 
                                                BELOW_MIN, 0, 0, 0.0, 0.0, 
                                                0, None), ), 
                                  nlgeom=ON)



def General_Contact(stepName):
    # Rename the two variables
    GC_IP = 'INTPROP-GC' # Interaction property
    GC = 'GENERAL_CONTACT' # General Contact name
    cIP = 'COHESIVE_INTPROP' # cohesive interaction property name
    abqModelPG.ContactProperty(GC_IP)
    
    GC_IntProp = abqModelPG.interactionProperties[GC_IP] # simplify code
    
    GC_IntProp.TangentialBehavior(formulation=PENALTY, 
                                  directionality=ISOTROPIC, 
                                  slipRateDependency=OFF, 
                                  pressureDependency=OFF, 
                                  temperatureDependency=OFF, 
                                  dependencies=0, 
                                  table=((0.2, ), ), 
                                  shearStressLimit=None, 
                                  maximumElasticSlip=FRACTION, 
                                  fraction=0.005, 
                                  elasticSlipStiffness=None)
    GC_IntProp.NormalBehavior(pressureOverclosure=HARD, 
                              allowSeparation=ON, 
                              constraintEnforcementMethod=DEFAULT)
    abqModelPG.ContactExp(name=GC, createStepName=stepName)
    
    GC_Int = abqModelPG.interactions[GC] # simplify code
    GC_Int.includedPairs.setValuesInStep(stepName=stepName, useAllstar=ON)
    GC_Int.contactPropertyAssignments.appendInStep(stepName=stepName, 
                                                   assignments=((GLOBAL, 
                                                                 SELF, 
                                                                 GC_IP), 
                                                                )
                                                   )
    
    if tieInterface == False:
        # INDIVIDUAL PROPERTY ASSIGNMENT
        master=abqModelPG.rootAssembly.instances['R-1'].surfaces['R_V_SURF_BOND']
        slave=abqModelPG.rootAssembly.instances['V-1'].surfaces['V_R_SURF_BOND']
        GC_Int.contactPropertyAssignments.appendInStep(stepName=stepName, 
                                                       assignments=((slave, 
                                                                     master, 
                                                                     cIP), 
                                                                    )
                                                       )
    else:
        print('Tied interface')



def PredefinedEyeHolderField(odbName):
    ''' Update the predefined state to the current model '''
    instances=(abqModelPG.rootAssembly.instances['E-1'], )
    abqModelPG.InitialState(updateReferenceConfiguration=ON, 
                            fileName=odbName, 
                            endStep=LAST_STEP, 
                            endIncrement=STEP_END, 
                            name='Predefined Field-EyeHolder', 
                            createStepName='Initial', 
                            instances=instances)



def PredefinedVitreousField(odbName):
    ''' Update the predefined state to the current model '''
    instances=(abqModelPG.rootAssembly.instances['V-1'], )
    abqModelPG.InitialState(updateReferenceConfiguration=ON, 
                            fileName=odbName, 
                            endStep=LAST_STEP, 
                            endIncrement=STEP_END, 
                            name='Predefined Field-Vitreous', 
                            createStepName='Initial', 
                            instances=instances)


def PredefinedRetinaField(odbName):
    ''' Update the predefined state to the current model '''
    instances=(abqModelPG.rootAssembly.instances['R-1'], )
    abqModelPG.InitialState(updateReferenceConfiguration=ON, 
                            fileName=odbName, 
                            endStep=LAST_STEP, 
                            endIncrement=STEP_END, 
                            name='Predefined Field-Retina', 
                            createStepName='Initial', 
                            instances=instances)


def Gravity(stepName):
    abqModelPG.Gravity(name='Gravity', createStepName=stepName, comp2=-9.81, 
                     distributionType=UNIFORM, field='')


def V_SYM_Constrain_BC(stepName):
    a = abqModelPG.rootAssembly
    f = a.instances['V-1'].faces
    faces = f.getSequenceFromMask(mask=('[#17000042 #6a ]', ), )
    region = a.Set(faces=faces, name='V_SYM_BC_SET')
    abqModel.ZsymmBC(name='V_sym', 
                     createStepName=stepName, 
                     region=region, 
                     localCsys=None)


def FEA_Post_Grav():
    PostGravity(odbFile)
    
    # Mat Props
    RetinaProp = 11120.0
    vitreousProp = 69.565490
    Retina_Mat_Prop(RetinaProp)
    Vitreous_Mat_Prop(vitreousProp)
    
    # Model specific features (Assign sections)
    E_Features()
    R_Features()
    V_Features()
    
    # Assembly
    Assembly_ODB()
    
    # Step
    stepName = 'Peel_Test_Dynamic_Explicit'
    descrip = 'Peel the retina away from the vitreous (rotational peel test)'
    Step_Post_Gravity(100, 100, 0.5**5, stepName, descrip)
    
    # Contact
    General_Contact(stepName)
    
    # Predefined fields
    PredefinedEyeHolderField(odbName)
    PredefinedVitreousField(odbName)
    PredefinedRetinaField(odbName)
    
    # Loads
    Gravity(stepName)
    
    
    # # # Boundary Conditions
    # V_SYM_Constrain_BC(stepName)
    # R_SYM_Constrain_BC(stepName)
    
    del mdb.models['Model-1']




if __name__ == '__main__':
    """ Run the following function """
    FEA_Post_Grav()