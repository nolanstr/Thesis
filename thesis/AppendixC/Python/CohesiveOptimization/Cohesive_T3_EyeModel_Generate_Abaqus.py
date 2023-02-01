# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 21:51:32 2021

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

# In[Non-symmetric model]
'''
Non-symmetric model
'''

# location of the folder
# specific folder path where this file is located # os.getcwd()
pythonScriptPath = os.path.abspath("file")
abqWD, pythonFiles = os.path.split(pythonScriptPath) # split file path

# StepFile = 'Adult Human Eye holder Assembly.STEP'
# StepFile = 'Adult Human Eye holder Assembly 2 Step.STEP'
# Constrained
# StepFile = 'Adult Human Eye holder Assembly Constrained Bottom.STEP'

# Trimmed to prevent element distortion on low elastic modulus curve fits
StepFile = ('Adult Human Eye holder Assembly Constrained Bottom Trimmed ' + 
            'Retina.STEP')

SolidWorksDir = 'SolidWorksStepFiles' # Folder name

# Combine folder directory
SolidWorksStepFile = os.path.join(SolidWorksDir, StepFile)

def ImportStepEyeConstrained():
    """ Use with the constrained bottom STEP file"""
    step = mdb.openStep(os.path.join(abqWD, SolidWorksStepFile), 
                        scaleFromFile=OFF)
    
    abqModel.PartFromGeometryFile(name='V', geometryFile=step, bodyNum=1, 
                                  combine=False, dimensionality=THREE_D, 
                                  type=DEFORMABLE_BODY)
    abqModel.PartFromGeometryFile(name='E', geometryFile=step, bodyNum=2, 
                                  combine=False, dimensionality=THREE_D, 
                                  type=DISCRETE_RIGID_SURFACE)
    abqModel.PartFromGeometryFile(name='R', geometryFile=step, bodyNum=3, 
                                  combine=False, dimensionality=THREE_D, 
                                  type=DEFORMABLE_BODY)
    abqModel.PartFromGeometryFile(name='T', geometryFile=step, bodyNum=4, 
                                  combine=False, dimensionality=THREE_D, 
                                  type=DISCRETE_RIGID_SURFACE)
    abqModel.PartFromGeometryFile(name='G', geometryFile=step, bodyNum=5, 
                                  combine=False, dimensionality=THREE_D, 
                                  type=DISCRETE_RIGID_SURFACE)


def Retina_Mat_Prop(RetinaProp):
    retina_E = RetinaProp # Passed in young's modulus
    Retina_Description = """
Actually used the value from Chen 2014
E = 11.12 KPa

------------------------------------------------
Density (kg/m^3)
1100 --------> Esposito_2013

"""
    abqModel.Material(name='Retina', description=Retina_Description)
    abqModel.materials['Retina'].Density(table=((1100.0, ), ))
    abqModel.materials['Retina'].Elastic(table=((retina_E, 0.49), ))
    
    # Assign the section to the part
    abqModel.HomogeneousSolidSection(name='Retina_Section', material='Retina', 
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
    abqModel.Material(name='Vitreous', description=Vitreous_Description)
    abqModel.materials['Vitreous'].Density(table=((950.0, ), ))
    ''' Using Lin2020 Paper to relate SLSM curve fit parameters to physical 
    values.  Prony 4 Term (Long term) initial guess 172.77874855377468 
    optimization of E'''
    abqModel.materials['Vitreous'].Elastic(moduli=LONG_TERM, 
                                           table=((vitreous_E, 0.49), ))
    # Prony 4 Term calculated from normalized data
    abqModel.materials['Vitreous'].Viscoelastic(
            domain=TIME, time=PRONY, table=(
            # Tram Data # 5
            (0.1486397420159951, 0.0, 331.4796231072498),
            (0.12469207412616717, 0.0, 3.388868494747128),
            (0.29059507092540404, 0.0, 15.59692349525066),
            (0.1591569334281, 0.0, 69.85134248442381)))
    
    # Assign the section to the part
    abqModel.HomogeneousSolidSection(name='Vitreous_Section', 
                                     material='Vitreous', thickness=None)


def E_Features():
    ''' Eye holder features '''
    p = abqModel.parts['E']
    
    # Remove shell
    c = p.cells
    p.RemoveCells(cellList = c[0:1])
    
    # Reference point
    p.ReferencePoint(point=(0.0, 0.0, 0.0))
    
    # Add E-set to the reference point
    r = p.referencePoints
    refPoints=(r[3], )
    p.Set(referencePoints=refPoints, name='E_RP_Set')
    
    # Edge seed sets
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#400f000 #1402 ]', ), )
    p.Set(edges=edges, name='E_Edge_Seed_Set')
    
    edges = e.getSequenceFromMask(mask=('[#f1ff0fff #2838 ]', ), )
    p.Set(edges=edges, name='E_Outside_Edge_Seed_Set')
    
    # Surfaces
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#1ffff ]', ), )
    p.Surface(side1Faces=side1Faces, name='E_Surf')


def G_Features():
    ''' Glue features '''
    p = abqModel.parts['G']
    c = p.cells
    
    # Remeove cells for rigid body
    p.RemoveCells(cellList = c[0:1])
    
    # Reference point
    p.ReferencePoint(point=(9.799E-03, 5.657E-03, 2.54E-03))
    
    # Define the reference point for the rigid body
    r = p.referencePoints
    refPoints=(r[3], )
    p.Set(referencePoints=refPoints, name='G_RP_Set')
    
    # # Create sets
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#3f ]', ), )
    p.Set(faces=faces, name='G_Set')
    faces = f.getSequenceFromMask(mask=('[#20 ]', ), )
    p.Set(faces=faces, name='G_T_Set')
    faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
    p.Set(faces=faces, name='G_R_Set')
    
    # Create surfaces
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#3f ]', ), )
    p.Surface(side1Faces=side1Faces, name='G_Surf')
    side1Faces = s.getSequenceFromMask(mask=('[#20 ]', ), )
    p.Surface(side1Faces=side1Faces, name='G_T_Surf')
    side1Faces = s.getSequenceFromMask(mask=('[#1 ]', ), )
    p.Surface(side1Faces=side1Faces, name='G_R_Surf')


def T_Features():
    ''' Plastic Tab features '''
    p = abqModel.parts['T']
    c = p.cells
    
    # Remeove cells for rigid body
    p.RemoveCells(cellList = c[0:1])
    
    # Reference point
    p.ReferencePoint(point=(16.241E-03, 9.74E-03, 13.E-06))
    
    # Define the reference point for the rigid body
    r = p.referencePoints
    refPoints=(r[3], )
    p.Set(referencePoints=refPoints, name='T_RP_Set')
    
    # Create sets
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#ff ]', ), )
    p.Set(faces=faces, name='T_Set')
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#2 ]', ), )
    p.Set(faces=faces, name='T_G_Set')
    
    # Create surfaces
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#ff ]', ), )
    p.Surface(side1Faces=side1Faces, name='T_Surf')
    side1Faces = s.getSequenceFromMask(mask=('[#2 ]', ), )
    p.Surface(side1Faces=side1Faces, name='T_G_Surf')


def R_Features():
    ''' Retina features '''
    p = abqModel.parts['R']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    p.Set(cells=cells, name='R_Set')
    
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#3 ]', ), )
    p.Set(faces=faces, name='R_G_Set')
    
    faces = f.getSequenceFromMask(mask=('[#4 ]', ), )
    p.Set(faces=faces, name='R_V_Set')
    
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#ff ]', ), )
    p.Surface(side1Faces=side1Faces, name='R_Surf')
    
    side1Faces = s.getSequenceFromMask(mask=('[#3 ]', ), )
    p.Surface(side1Faces=side1Faces, name='R_G_Surf')
    
    side1Faces = s.getSequenceFromMask(mask=('[#4 ]', ), )
    p.Surface(side1Faces=side1Faces, name='R_V_Surf_BOND')
    
    # Assign section
    region = p.sets['R_Set']
    p.SectionAssignment(region=region, sectionName='Retina_Section', 
                        offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', 
                        thicknessAssignment=FROM_SECTION)



def PartitionRetinaOnVitreous():
    ''' Vitreous features additional partitions for creating the surface for 
    bonding'''
    p = abqModel.parts['V']
    
    # Partition V along the width of the retina
    p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=-0.00254)
    abqModel.parts['V'].features.changeKey(fromName='Datum plane-1', 
                                           toName='Retina_Width_Neg_Z')
    
    p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=0.00254)
    abqModel.parts['V'].features.changeKey(fromName='Datum plane-1', 
                                           toName='Retina_Width_Pos_Z')
    
    # Create a datum plnd along the z axis plane
    p.DatumAxisByPrincipalAxis(principalAxis=ZAXIS)
    p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=0.0)
    
    # Create the rotated datum planes
    d = p.datums
    p.DatumPlaneByRotation(plane=d[5], axis=d[4], angle=18.75)
    p.DatumPlaneByRotation(plane=d[5], axis=d[4], angle=-18.75)
    
    ''' Partition the surface of the retina on the vitreous '''
    p = abqModel.parts['V']
    c, d = p.cells, p.datums
    pickedCells = c.getSequenceFromMask(mask=('[#440 ]', ), )
    p.PartitionCellByDatumPlane(datumPlane=d[3], cells=pickedCells)
    pickedCells = c.getSequenceFromMask(mask=('[#408 ]', ), )
    p.PartitionCellByDatumPlane(datumPlane=d[2], cells=pickedCells)
    pickedCells = c.getSequenceFromMask(mask=('[#22 ]', ), )
    p.PartitionCellByDatumPlane(datumPlane=d[6], cells=pickedCells)
    pickedCells = c.getSequenceFromMask(mask=('[#140 ]', ), )
    p.PartitionCellByDatumPlane(datumPlane=d[7], cells=pickedCells)


def Vitreous_Features():
    ''' Assign specific features to the vitreous '''
    p = abqModel.parts['V']
    c, f, s = p.cells, p.faces, p.faces
    
    # Sets
    cells = c.getSequenceFromMask(mask=('[#ffffff ]', ), )
    p.Set(cells=cells, name='V_Set')
    faces = f.getSequenceFromMask(mask=('[#5090 ]', ), )
    p.Set(faces=faces, name='V_R_Set')
    
    # Surfaces
    side1Faces = s.getSequenceFromMask(mask=('[#1805090 #3 #ff0 ]', ), )
    p.Surface(side1Faces=side1Faces, name='V_Surf')
    side1Faces = s.getSequenceFromMask(mask=('[#5090 ]', ), )
    p.Surface(side1Faces=side1Faces, name='V_R_Surf_BOND')
    
    # Assign the section to the part
    region = p.sets['V_Set']
    p.SectionAssignment(region=region, 
                        sectionName='Vitreous_Section', 
                        offset=0.0, 
                        offsetType=MIDDLE_SURFACE, 
                        offsetField='', 
                        thicknessAssignment=FROM_SECTION)


def V_Partition_XYZ_Axis():
    ''' Partition the sphere along the x, y, z axis '''
    p = abqModel.parts['V']
    c, v, e, d = p.cells, p.vertices, p.edges, p.datums
    pickedCells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    p.PartitionCellByPlaneThreePoints(point1=v[1], 
                                      point2=v[0], 
                                      point3=v[3], 
                                      cells=pickedCells)
    
    pickedCells = c.getSequenceFromMask(mask=('[#3 ]', ), )
    p.PartitionCellByPlaneThreePoints(point1=v[0], 
                                      point2=v[4], 
                                      point3=v[2], 
                                      cells=pickedCells)
    
    pickedCells = c.getSequenceFromMask(mask=('[#f ]', ), )
    p.PartitionCellByPlaneThreePoints(point1=v[5], 
                                      point2=v[2], 
                                      point3=v[4], 
                                      cells=pickedCells)


def V_Internal_Sphere():
    sphereRadius = 0.008 # radius of the internal sphere for meshing
    
    s1 = abqModel.ConstrainedSketch(name='__profile__', sheetSize=0.1)
    g, v, d, c1 = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.sketchOptions.setValues(decimalPlaces=3)
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -0.05), point2=(0.0, 0.05))
    s1.FixedConstraint(entity=g[2])
    s1.ArcByCenterEnds(center=(0.0, 0.0), 
                        point1=(0.0, sphereRadius), 
                        point2=(0.0, -sphereRadius), 
                        direction=CLOCKWISE)
    s1.Line(point1=(0.0, sphereRadius), 
            point2=(0.0, -sphereRadius))
    s1.VerticalConstraint(entity=g[4], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
    p = abqModel.Part(name='V_internal', 
                      dimensionality=THREE_D, 
                      type=DEFORMABLE_BODY)
    p = abqModel.parts['V_internal']
    p.BaseSolidRevolve(sketch=s1, angle=360.0, flipRevolveDirection=OFF)
    s1.unsetPrimaryObject()
    p = abqModel.parts['V_internal']
    del abqModel.sketches['__profile__']


def mergeV():
    ''' Merge the internal sphere with the vitreous '''
    a = abqModel.rootAssembly
    a.InstanceFromBooleanMerge(name='V_Merge', 
                               instances=(a.instances['V-1'], 
                                          a.instances['V_internal-1'], ), 
                               keepIntersections=ON, 
                               originalInstances=DELETE, 
                               domain=GEOMETRY)
    
    # Clean up file names after merge
    del abqModel.parts['V']
    del abqModel.parts['V_internal']
    
    abqModel.parts.changeKey(fromName='V_Merge', toName='V')
    a = abqModel.rootAssembly
    a.regenerate()
    abqModel.rootAssembly.features.changeKey(fromName='V_Merge-1', 
                                              toName='V-1')
    
    a.regenerate()


def AssembleV_for_Merging():
    a1 = abqModel.rootAssembly
    a1.DatumCsysByDefault(CARTESIAN)
    p = abqModel.parts['V']
    a1.Instance(name='V-1', part=p, dependent=ON)
    p = abqModel.parts['V_internal']
    a1.Instance(name='V_internal-1', part=p, dependent=ON)


def E_Mesh(InsideSeed, OutsideSeed):
    p = abqModel.parts['E']
    e = p.edges
    pickedEdges = e.getSequenceFromMask(mask=('[#400f000 #1402 ]', ), )
    p.seedEdgeBySize(edges=pickedEdges, 
                     size=0.0005, 
                     deviationFactor=0.1, 
                     minSizeFactor=0.1, 
                     constraint=FINER)
    pickedEdges = e.getSequenceFromMask(mask=('[#f1ff0fff #2838 ]', ), )
    p.seedEdgeBySize(edges=pickedEdges, 
                     size=0.00342673, 
                     deviationFactor=0.1, 
                     minSizeFactor=0.1, 
                     constraint=FINER)
    # (unique node numbering)
    p.setValues(startNodeLabel=1000000, startElemLabel=1000000)
    p.generateMesh()


def G_Mesh(seed):
    p = abqModel.parts['G']
    p.seedPart(size=seed, deviationFactor=0.1, minSizeFactor=0.1)
    f = p.faces
    pickedRegions = f.getSequenceFromMask(mask=('[#3f ]', ), )
    p.setMeshControls(regions=pickedRegions, elemShape=QUAD)
    elemType1 = mesh.ElemType(elemCode=R3D4, elemLibrary=EXPLICIT)
    elemType2 = mesh.ElemType(elemCode=R3D3, elemLibrary=EXPLICIT)
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#3f ]', ), )
    pickedRegions =(faces, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
    # (unique node numbering)
    p.setValues(startNodeLabel=2000000, startElemLabel=2000000)
    p.generateMesh()


def T_Mesh(seed):
    p = abqModel.parts['T']
    p.seedPart(size=seed, deviationFactor=0.1, minSizeFactor=0.1)
    f = p.faces
    pickedRegions = f.getSequenceFromMask(mask=('[#ff ]', ), )
    p.setMeshControls(regions=pickedRegions, elemShape=QUAD)
    elemType1 = mesh.ElemType(elemCode=R3D4, elemLibrary=EXPLICIT)
    elemType2 = mesh.ElemType(elemCode=R3D3, elemLibrary=EXPLICIT)
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#ff ]', ), )
    pickedRegions =(faces, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
    # (unique node numbering)
    p.setValues(startNodeLabel=3000000, startElemLabel=3000000)
    p.generateMesh()


def R_Mesh(seed):
    p = abqModel.parts['R']
    p.seedPart(size=seed, deviationFactor=0.1, minSizeFactor=0.1)
    c, e = p.cells, p.edges
    pickedRegions = c.getSequenceFromMask(mask=('[#1 ]', ), )
    p.setMeshControls(regions=pickedRegions, 
                      technique=SWEEP, 
                      algorithm=ADVANCING_FRONT)
    p.setSweepPath(region=c[0], edge=e[10], sense=FORWARD)
    elemType1 = mesh.ElemType(elemCode=C3D8R, 
                              elemLibrary=EXPLICIT, 
                              kinematicSplit=AVERAGE_STRAIN, 
                              secondOrderAccuracy=ON, 
                              hourglassControl=ENHANCED, 
                              distortionControl=ON, 
                              lengthRatio=0.100000001490116)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=EXPLICIT)
    elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=EXPLICIT)
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    pickedRegions =(cells, )
    p.setElementType(regions=pickedRegions, 
                     elemTypes=(elemType1, elemType2, elemType3))
    p.generateMesh()
    # (unique node numbering)
    p.setValues(startNodeLabel=4000000, startElemLabel=4000000)
    p.generateMesh()


def VitreousMesh(v1Seed, v2Seed):
    ''' Specity tetrahedral elements '''
    p = abqModel.parts['V']
    c = p.cells
    pickedRegions = c.getSequenceFromMask(mask=('[#86f800 ]', ), )
    p.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE)
    elemType1 = mesh.ElemType(elemCode=C3D20R)
    elemType2 = mesh.ElemType(elemCode=C3D15)
    elemType3 = mesh.ElemType(elemCode=C3D10)
    cells = c.getSequenceFromMask(mask=('[#86f800 ]', ), )
    pickedRegions =(cells, )
    p.setElementType(regions=pickedRegions, 
                     elemTypes=(elemType1, elemType2, elemType3))
    
    ''' Specify hexahedral elements '''
    elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=EXPLICIT)
    elemType3 = mesh.ElemType(elemCode=C3D4, 
                              elemLibrary=EXPLICIT, 
                              secondOrderAccuracy=ON, 
                              distortionControl=ON, 
                              lengthRatio=0.100000001490116)
    cells = c.getSequenceFromMask(mask=('[#86f800 ]', ), )
    pickedRegions =(cells, )
    p.setElementType(regions=pickedRegions, 
                     elemTypes=(elemType1, elemType2, elemType3))
    
    elemType1 = mesh.ElemType(elemCode=C3D8R, 
                              elemLibrary=EXPLICIT, 
                              kinematicSplit=AVERAGE_STRAIN, 
                              secondOrderAccuracy=ON, 
                              hourglassControl=ENHANCED, 
                              distortionControl=ON, 
                              lengthRatio=0.100000001490116)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=EXPLICIT)
    elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=EXPLICIT)
    cells = c.getSequenceFromMask(mask=('[#7907ff ]', ), )
    pickedRegions =(cells, )
    p.setElementType(regions=pickedRegions, 
                     elemTypes=(elemType1, elemType2, elemType3))
    
    # Seed the entire part
    p.seedPart(size=v2Seed, deviationFactor=0.1, minSizeFactor=0.1)
    
    # Seed the retina interface
    e = p.edges
    pickedEdges = e.getSequenceFromMask(mask=('[#ffffffff #7fec0fff #80012 ]',
                                              ), )
    p.seedEdgeBySize(edges=pickedEdges, 
                     size=v1Seed, 
                     deviationFactor=0.1, 
                     minSizeFactor=0.1, 
                     constraint=FINER)
    
    # Seed the bias edges
    e = p.edges
    pickedEdges1 = e.getSequenceFromMask(mask=('[#0 #104000 #10001 ]', ), )
    pickedEdges2 = e.getSequenceFromMask(mask=('[#0 #80020000 #900000 ]', ), )
    p.seedEdgeByBias(biasMethod=SINGLE, 
                     end1Edges=pickedEdges1, 
                     end2Edges=pickedEdges2, 
                     minSize=v1Seed, 
                     maxSize=v2Seed, 
                     constraint=FINER)
    
    # (unique node numbering)
    p.setValues(startNodeLabel=5000000, startElemLabel=5000000)
    p.generateMesh()


def QuadraticTetVitreous():
    # Vitreous
    p = abqModel.parts['V']
    c = p.cells
    pickedRegions = c.getSequenceFromMask(mask=('[#9f ]', ), )
    p.deleteMesh(regions=pickedRegions)
    p.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE)
    elemType1 = mesh.ElemType(elemCode=UNKNOWN_HEX, elemLibrary=EXPLICIT)
    elemType2 = mesh.ElemType(elemCode=UNKNOWN_WEDGE, elemLibrary=EXPLICIT)
    elemType3 = mesh.ElemType(elemCode=C3D10M, elemLibrary=EXPLICIT)
    cells = c.getSequenceFromMask(mask=('[#9f ]', ), )
    pickedRegions =(cells, )
    p.setElementType(regions=pickedRegions, 
                     elemTypes=(elemType1, elemType2, elemType3))
    elemType1 = mesh.ElemType(elemCode=UNKNOWN_HEX, elemLibrary=EXPLICIT)
    elemType2 = mesh.ElemType(elemCode=UNKNOWN_WEDGE, elemLibrary=EXPLICIT)
    elemType3 = mesh.ElemType(elemCode=C3D10M, 
                              elemLibrary=EXPLICIT, 
                              secondOrderAccuracy=ON, 
                              distortionControl=ON, 
                              lengthRatio=0.100000001490116)
    c = p.cells
    p.setElementType(regions=pickedRegions, 
                     elemTypes=(elemType1, elemType2, elemType3))
    p.generateMesh()


def QuadraticTetRetina():
    # Retina
    p = abqModel.parts['R']
    c = p.cells
    pickedRegions = c.getSequenceFromMask(mask=('[#1 ]', ), )
    p.deleteMesh(regions=pickedRegions)
    p.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE)
    elemType1 = mesh.ElemType(elemCode=UNKNOWN_HEX, elemLibrary=EXPLICIT)
    elemType2 = mesh.ElemType(elemCode=UNKNOWN_WEDGE, elemLibrary=EXPLICIT)
    elemType3 = mesh.ElemType(elemCode=C3D10M, elemLibrary=EXPLICIT)
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    pickedRegions =(cells, )
    p.setElementType(regions=pickedRegions, 
                      elemTypes=(elemType1, elemType2, elemType3))
    elemType1 = mesh.ElemType(elemCode=UNKNOWN_HEX, elemLibrary=EXPLICIT)
    elemType2 = mesh.ElemType(elemCode=UNKNOWN_WEDGE, elemLibrary=EXPLICIT)
    elemType3 = mesh.ElemType(elemCode=C3D10M, 
                              elemLibrary=EXPLICIT, 
                              secondOrderAccuracy=ON, 
                              distortionControl=ON, 
                              lengthRatio=0.100000001490116)
    p.setElementType(regions=pickedRegions, 
                      elemTypes=(elemType1, elemType2, elemType3))
    p.generateMesh()


def Assembly():
    a1 = abqModel.rootAssembly
    a1.DatumCsysByDefault(CARTESIAN)
    p = abqModel.parts['E']
    a1.Instance(name='E-1', part=p, dependent=ON)
    p = abqModel.parts['G']
    a1.Instance(name='G-1', part=p, dependent=ON)
    p = abqModel.parts['R']
    a1.Instance(name='R-1', part=p, dependent=ON)
    p = abqModel.parts['T']
    a1.Instance(name='T-1', part=p, dependent=ON)
    p = abqModel.parts['V']
    a1.Instance(name='V-1', part=p, dependent=ON)


def GravityStep(time, prevStep, scaleFactor, MSTI, stepName, descrip):
    abqModel.ExplicitDynamicsStep(name=stepName, 
                                  previous=prevStep, 
                                  description=descrip, 
                                  timePeriod=time, 
                                  massScaling=((SEMI_AUTOMATIC, 
                                                MODEL, 
                                                AT_BEGINNING, 
                                                scaleFactor, 
                                                MSTI, 
                                                BELOW_MIN, 0, 0, 0.0, 0.0, 0, 
                                                None), ))


def Step(time, prevStep, scaleFactor, MSTI, stepName, descrip):
    abqModel.ExplicitDynamicsStep(name=stepName, 
                                  previous=prevStep, 
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
    # Mass Scale default
    if MSTI == 0:
        print('This will take a while...ABAQUS is deciding for us')
        # Use zero value
        abqModel.steps[stepName].setValues(massScaling=PREVIOUS_STEP)
    else:
        print('Mass Scale Time Increment has been defined')


def F_output(stepName):
    FOutputInterval = 50 # Double the data points (Default is 20)
    
    if damageInitiation == False and damageEvolution == False:
        # Whole Model Fieldoutput (RF, U, NFORC)
        abqModel.FieldOutputRequest(name='F-Output-1', 
                                    createStepName=stepName, 
                                    variables=('RF', 
                                               'U', 
                                               'NFORC'), 
                                    numIntervals=FOutputInterval)
    
    elif damageInitiation == True and damageEvolution == False:
        # Whole Model Fieldoutput (RF, U, NFORC)
        abqModel.FieldOutputRequest(name='F-Output-1', 
                                    createStepName=stepName, 
                                    variables=('RF', 
                                               'U', 
                                               'NFORC', 
                                               'CSMAXSCRT'), 
                                    numIntervals=FOutputInterval)
    
    elif damageInitiation == True and damageEvolution == True:
        # Whole Model Fieldoutput (RF, U, NFORC)
        abqModel.FieldOutputRequest(name='F-Output-1', 
                                    createStepName=stepName, 
                                    variables=('RF', 
                                               'U', 
                                               'NFORC', 
                                               'CSDMG', 
                                               'CSMAXSCRT'), 
                                    numIntervals=FOutputInterval)
    
    # Set specific field output (Retina LE & S)
    regionDef=abqModel.rootAssembly.allInstances['R-1'].sets['R_Set']
    abqModel.FieldOutputRequest(name='Retina_LE_S', 
                                createStepName=stepName, 
                                variables=('LE', 
                                           
                                           'S'), 
                                numIntervals=FOutputInterval, 
                                region=regionDef, 
                                sectionPoints=DEFAULT, 
                                rebar=EXCLUDE)
    
    # Set specific field output (Vitreous LE & S)
    regionDef=abqModel.rootAssembly.allInstances['V-1'].sets['V_Set']
    abqModel.FieldOutputRequest(name='Vitreous_LE_S', 
                                createStepName=stepName, 
                                variables=('LE', 
                                           'S'), 
                                numIntervals=FOutputInterval, 
                                region=regionDef, 
                                sectionPoints=DEFAULT, 
                                rebar=EXCLUDE)
    
    # # Set specific field output (Rigid Body U & RF)
    # regionDef=abqModel.rootAssembly.allInstances['G-1'].sets['G_RP_Set']
    # abqModel.FieldOutputRequest(name='Glue_U_RF', 
    #                             createStepName=stepName, 
    #                             variables=('U', 
    #                                        'RF'), 
    #                             numIntervals=FOutputInterval, 
    #                             region=regionDef, 
    #                             sectionPoints=DEFAULT, 
    #                             rebar=EXCLUDE)


def H_output(stepName):
    # Internal/Kinetic Energy
    abqModel.HistoryOutputRequest(name='H-Output-1', 
                                  createStepName=stepName, 
                                  variables=('ALLIE', 
                                             'ALLKE'))
    
    # # Define specific reaction force on the glue reference point
    # a = abqModel.rootAssembly
    # regionDef=a.allInstances['G-1'].sets['G_RP_Set']
    # abqModel.HistoryOutputRequest(name='G_RP_Output_U_RF_RM', 
    #                               createStepName=stepName, 
    #                               variables=('U1', 'U2', 'U3', 
    #                                          'RF1', 'RF2', 'RF3', 
    #                                          'RM1', 'RM2', 'RM3'), 
    #                               region=regionDef, sectionPoints=DEFAULT, 
    #                               rebar=EXCLUDE)


def General_Contact(stepName, cIP):
    # Rename the two variables
    GC_IP = 'IntProp-GC' # Interaction property
    GC = 'General_Contact' # General Contact name
    # cIP = 'cohesive_IntProp' # cohesive interaction property name
    abqModel.ContactProperty(GC_IP)
    
    GC_IntProp = abqModel.interactionProperties[GC_IP] # simplify code
    
    # if gravity == True:
    #     # Gravity keeps the vitreous from energetically moving after peeling
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
    # else:
    #     # Prevent the vitreous from sliding inside the eye holder
    #     GC_IntProp.TangentialBehavior(formulation=ROUGH)
    
    GC_IntProp.NormalBehavior(pressureOverclosure=HARD, 
                              allowSeparation=ON, 
                              constraintEnforcementMethod=DEFAULT)
    abqModel.ContactExp(name=GC, createStepName=stepName)
    
    GC_Int = abqModel.interactions[GC] # simplify code
    GC_Int.includedPairs.setValuesInStep(stepName=stepName, useAllstar=ON)
    GC_Int.contactPropertyAssignments.appendInStep(stepName=stepName, 
                                                   assignments=((GLOBAL, 
                                                                 SELF, 
                                                                 GC_IP), 
                                                                )
                                                   )


def updateGeneralContact(stepName, Knn, Kss, Ktt, damageInitiation, 
                         tn, ts, tt, damageEvolution, FE):
    ''' Specify the cohesive surface behavior between the retina and vitreous
    during the step after the gravity step '''
    # Simplify
    GC = 'General_Contact'
    cp = 'cohesivePeel'
    
    abqModel.ContactProperty(cp)
    
    CP_IP = abqModel.interactionProperties[cp]
    CP_IP.TangentialBehavior(formulation=PENALTY, 
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
    
    CP_IP.CohesiveBehavior(defaultPenalties=OFF, 
                           table=((Knn, Kss, Ktt), ))
    # eligibility=INITIAL_NODES, 
    
    CP_IP.Damage(criterion=MAX_STRESS, 
                 initTable=((tn, ts, tt), ), 
                 useEvolution=ON, 
                 evolutionType=ENERGY, 
                 evolTable=((FE, ), ), 
                 useStabilization=ON, 
                 viscosityCoef=1e-05)
    
    GCI = abqModel.interactions[GC]
    if gravity == True:
        GCI.contactPropertyAssignments.changeValuesInStep(stepName=stepName, 
                                                          index=1, 
                                                          value=cp)
    else:
        r11=abqModel.rootAssembly.instances['R-1'].surfaces['R_V_Surf_BOND']
        r12=abqModel.rootAssembly.instances['V-1'].surfaces['V_R_Surf_BOND']
        GCI.contactPropertyAssignments.appendInStep(stepName=stepName, 
                                                    assignments=((r11, r12, 
                                                                  cp), ))


def smoothGravity():
    abqModel.SmoothStepAmplitude(name='smoothGravity', timeSpan=STEP, 
        data=((0.0, 0.0), (100.0, 1.0)))
    abqModel.loads['Gravity'].setValues(amplitude='smoothGravity', 
        distributionType=UNIFORM, field='')


def turnTieCohesive(stepName, cohTieName):
    ''' Simulate the tie constraint with cohesive surface '''
    GC = 'General_Contact'
    CTG = cohTieName # Simplify
    abqModel.ContactProperty(CTG)
    
    # Simplify
    CTG_IP = abqModel.interactionProperties[CTG]
    GC_IP = abqModel.interactions[GC]
    
    CTG_IP.CohesiveBehavior(eligibility=INITIAL_NODES)
    r11=abqModel.rootAssembly.instances['R-1'].surfaces['R_V_Surf_BOND']
    r12=abqModel.rootAssembly.instances['V-1'].surfaces['V_R_Surf_BOND']
    GC_IP.contactPropertyAssignments.appendInStep(stepName=stepName, 
                                                  assignments=((r11, 
                                                                r12, 
                                                                CTG), ))


def peelStepPostGravity(time, stepName, previousStep, descrip, scaleFactor, 
                        MSTI):
    ''' step after the gravity phase '''
    abqModel.ExplicitDynamicsStep(name=stepName, 
                                  previous=previousStep, 
                                  description=descrip, 
                                  timePeriod=time, 
                                  massScaling=((SEMI_AUTOMATIC, 
                                                MODEL, AT_BEGINNING, 
                                                scaleFactor, MSTI, BELOW_MIN, 
                                                0, 0, 0.0, 0.0, 0, None), ))


def peelTestBCUpdate_With_Gravity(stepName):
    ''' Update the boundary conditions in the second step, post gravity '''
    abqModel.loads['Gravity'].setValuesInStep(stepName=stepName, 
                                              amplitude=FREED)
    abqModel.boundaryConditions['EHR'].setValuesInStep(stepName=stepName, 
                                                       vr3=-1.0)


def RG_Tie():
    a = abqModel.rootAssembly
    region1=a.instances['G-1'].surfaces['G_R_Surf']
    a = abqModel.rootAssembly
    region2=a.instances['R-1'].surfaces['R_G_Surf']
    abqModel.Tie(name='RG', 
                 master=region2, 
                 slave=region1, 
                 positionToleranceMethod=COMPUTED, 
                 adjust=OFF, 
                 tieRotations=ON, 
                 constraintEnforcement=SURFACE_TO_SURFACE, 
                 thickness=ON)


def Amp():
    abqModel.SmoothStepAmplitude(name='TD_amp', timeSpan=STEP, 
                                 data=((0.0, 0.0), 
                                       (30.0, 2e-05))
                                 )
    abqModel.SmoothStepAmplitude(name='omega', timeSpan=STEP, 
                                 data=((0.0, 0.0), 
                                       (30.0, 0.000909174))
                                 )


def EHR_BC_Fixed(stepName):
    a = abqModel.rootAssembly
    region = a.instances['E-1'].sets['E_RP_Set']
    abqModel.VelocityBC(name='EHR', createStepName=stepName, region=region, 
                        v1=0.0, v2=0.0, v3=0.0, vr1=0.0, vr2=0.0, vr3=0.0, 
                        amplitude='omega', localCsys=None, 
                        distributionType=UNIFORM, fieldName='')


def EHR_BC(stepName):
    a = abqModel.rootAssembly
    region = a.instances['E-1'].sets['E_RP_Set']
    abqModel.VelocityBC(name='EHR', createStepName=stepName, region=region, 
                        v1=0.0, v2=0.0, v3=0.0, vr1=0.0, vr2=0.0, vr3=-1.0, 
                        amplitude='omega', localCsys=None, 
                        distributionType=UNIFORM, fieldName='')

def GD_BC(stepName):
    a = abqModel.rootAssembly
    region = a.instances['G-1'].sets['G_RP_Set']
    abqModel.VelocityBC(name='GD', createStepName=stepName, 
                        region=region, v1=0.866092, v2=0.499884, v3=0.0, 
                        vr1=0.0, vr2=0.0, vr3=0.0, amplitude='TD_amp', 
                        localCsys=None, distributionType=UNIFORM, 
                        fieldName='')


def TD_BC(stepName):
    a = abqModel.rootAssembly
    region = a.instances['T-1'].sets['T_RP_Set']
    abqModel.VelocityBC(name='TD', createStepName=stepName, region=region, 
                        v1=0.866092, v2=0.499884, v3=0.0, 
                        vr1=0.0, vr2=0.0, vr3=0.0, amplitude='TD_amp', 
                        localCsys=None, distributionType=UNIFORM, 
                        fieldName='')


def Retina_Disp_BC(stepName):
    a = abqModel.rootAssembly
    region = a.instances['R-1'].sets['R_G_Set']
    abqModel.VelocityBC(name='R_Vel', 
                        createStepName=stepName, 
                        region=region, 
                        v1=0.866092, 
                        v2=0.499884, 
                        v3=UNSET, 
                        vr1=UNSET, 
                        vr2=UNSET, 
                        vr3=UNSET, 
                        amplitude='TD_amp', 
                        localCsys=None, 
                        distributionType=UNIFORM, 
                        fieldName='')

def Gravity(stepName):
    abqModel.Gravity(name='Gravity', createStepName=stepName, comp2=-9.81, 
                     distributionType=UNIFORM, field='')


def Write_Job(jobName, modelName, jobDescription):
    mdb.Job(name=jobName, 
            model=modelName, 
            description=jobDescription, 
            type=ANALYSIS, 
            atTime=None, 
            waitMinutes=0, 
            waitHours=0, 
            queue=None, 
            memory=90, 
            memoryUnits=PERCENTAGE, 
            explicitPrecision=DOUBLE, 
            nodalOutputPrecision=SINGLE, 
            echoPrint=OFF, 
            modelPrint=OFF, 
            contactPrint=OFF, 
            historyPrint=OFF, 
            userSubroutine='', 
            scratch='', 
            resultsFormat=ODB, 
            parallelizationMethodExplicit=DOMAIN, 
            numDomains=14, 
            activateLoadBalancing=False, 
            multiprocessingMode=DEFAULT, 
            numCpus=14)


def Save_INP(jobName):
    mdb.jobs[jobName].writeInput(consistencyChecking=OFF)


def VR_Tie():
    a = abqModel.rootAssembly
    slaveSurf=a.instances['V-1'].surfaces['V_R_Surf_BOND']
    mastSurf=a.instances['R-1'].surfaces['R_V_Surf_BOND']
    abqModel.Tie(name='RV_Tie', 
                 master=mastSurf, 
                 slave=slaveSurf, 
                 positionToleranceMethod=COMPUTED, 
                 adjust=OFF, 
                 tieRotations=ON, 
                 constraintEnforcement=SURFACE_TO_SURFACE, 
                 thickness=ON)
    return '_VR_Tie'


def JobNameFile(modelName, fileNameAttributes, jobDescription):
    """
    Creates a txt file with the jobNames and all attributes associated with 
    the model
    """
    fileName = modelName + fileNameAttributes
    outfile = open(fileName+'.txt', 'w')
    line = ('The file name indicates what parameters were used to define ' + 
            'the model\n')
    outfile.write(line)
    line = '\n' + fileName + '\n'
    outfile.write(line)
    line = jobDescription
    outfile.write(line)
    outfile.close()


def Submit_job(jobname):
    myJob = mdb.jobs[jobname]
    try:
        myJob.submit(consistencyChecking=OFF)
        myJob.waitForCompletion()
    except:
        print(str(datetime.datetime.now())+' stop by error!')
        pass


def RemoveRigid(stepName):
    """ Remove the rigid bodies all together """
    a = abqModel.rootAssembly
    a.features['T-1'].suppress()
    a.features['G-1'].suppress()
    abqModel.fieldOutputRequests['Glue_U_RF'].suppress()
    # abqModel.historyOutputRequests['Contact_CP-R-G'].suppress()
    abqModel.historyOutputRequests['G_RP_Output_U_RF_RM'].suppress()
    # abqModel.interactions['CP-R-G'].suppress()
    # abqModel.constraints['RG'].suppress()
    abqModel.boundaryConditions['GD'].suppress()
    abqModel.boundaryConditions['TD'].suppress()
    r11=a.instances['E-1'].surfaces['E_Surf']
    r12=a.instances['T-1'].surfaces['T_Surf']
    r21=a.instances['E-1'].surfaces['E_Surf']
    r22=a.instances['G-1'].surfaces['G_Surf']
    r31=a.instances['G-1'].surfaces['G_Surf']
    r32=a.instances['T-1'].surfaces['T_Surf']
    
    GC = 'General_Contact' # simplify
    GCI = abqModel.interactions[GC]
    GCI.excludedPairs.setValuesInStep(stepName=stepName, 
                                      removePairs=((r11, r12), 
                                                   (r21, r22), 
                                                   (r31, r32)))
    region = a.instances['R-1'].sets['R_G_Set']
    abqModel.VelocityBC(name='R_Vel', 
                        createStepName=stepName, 
                        region=region, 
                        v1=0.866092, 
                        v2=0.499884, 
                        v3=UNSET, 
                        vr1=UNSET, 
                        vr2=UNSET, 
                        vr3=UNSET, 
                        amplitude='TD_amp', 
                        localCsys=None, 
                        distributionType=UNIFORM, 
                        fieldName='')


def CohesiveSurface(Knn, Kss, Ktt, damageInitiation, tn, ts, tt,
                    damageEvolution, FE):
    coh_int_prop = 'cohesive_IntProp' # simplify
    abqModel.ContactProperty(coh_int_prop)
    C_IP = abqModel.interactionProperties[coh_int_prop] # simplify code
    
    C_IP.TangentialBehavior(formulation=PENALTY, 
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
    
    C_IP.CohesiveBehavior(defaultPenalties=OFF, 
                          table=((Knn, 
                                  Kss, 
                                  Ktt), ))
    
    if damageInitiation == True and damageEvolution == True:
        # If damage initiation and evolution are both turned on
        C_IP.Damage(criterion=MAX_STRESS, 
                    initTable=((tn, 
                                ts, 
                                tt), ), 
                    useEvolution=ON, 
                    evolutionType=ENERGY, 
                    softening=LINEAR, 
                    evolTable=((FE, ), ), 
                    useStabilization=ON, 
                    viscosityCoef=1e-5) # was EXPONENTIAL, LINEAR
        
    elif damageInitiation == True and damageEvolution == False:
        # If damage initation is turned on but evolution is not
        C_IP.Damage(criterion=MAX_STRESS, 
                    initTable=((tn, 
                                ts, 
                                tt), ), 
                    useEvolution=OFF, 
                    useStabilization=OFF)
    else:
        print('No damage initiation or evolution')


def FEA():
    """
    Function that generates FEA code to model vitreoretinal adhesion
    
    # Steps are as follows:
        1 - Create new model database
        2 - Import SolidWorks STEP file (Includes all parts)
        3 - Material property definitions
        4 - Part features (Element & Node Sets & Reference Points ...)
        5 - Mesh parts (Specify seed size)
        6 - Assembly
        7 - Step (Dynamic Explicit with Mass Scaling)
        8 - Outputs (Field & History)
        9 - Contact (General Contact)
        10 - Contact pair (Retina/Vitreous - Cohesive Surface)
        11 - Tie Constraint (Retina - Glue)
        12 - Amplitude definition
        13 - BC's'
        14 - Submit Job  :)
    """
    
    # Import SolidWorks STEP file
    ImportStepEyeConstrained()
    
    # Mat Props
    Retina_Mat_Prop(RetinaProp)
    Vitreous_Mat_Prop(VitreousProp)
    
    # # Part Geometry/RPs/Sets/Surfaces
    E_Features()
    G_Features()
    T_Features()
    R_Features()
    
    # Internal sphere to reduce mesh
    V_Partition_XYZ_Axis()
    V_Internal_Sphere()
    AssembleV_for_Merging()
    mergeV()
    
    # Features on the vitreous
    PartitionRetinaOnVitreous()
    Vitreous_Features()
    
    # Seed & Mesh parts
    E_Mesh(e1Seed, e2Seed) # Max/min
    G_Mesh(gSeed)
    T_Mesh(ptSeed)
    R_Mesh(rSeed)
    VitreousMesh(v1Seed, v2Seed)
    
    # Assembly
    Assembly()
    
    # Eliminate the glue and tab from the model
    a = abqModel.rootAssembly
    a.features['G-1'].suppress()
    a.features['T-1'].suppress()
    
    # Gravity Step
    previousStep = 'Initial'
    if gravity == True:
        stepName = 'Gravity_Step'
        descrip = ('Applying gravity to the model and letting the ' + 
                   'vitreous and retina settle')
        
        GravityStep(200, previousStep, scaleFactor, 0.03125, stepName, descrip)
        Gravity(stepName)
        smoothGravity()
        
        # Interactions
        cohTieName = 'Cohesive_Gravity_Tie'
        General_Contact(stepName, cohTieName)
        
        # Interaction properties
        turnTieCohesive(stepName, cohTieName)
        
        # Zero movement boundary conditions
        Amp()
        EHR_BC_Fixed(stepName)
        
        # # Model outputs for gravity step
        F_output(stepName)
        H_output(stepName)
        
        previousStep = stepName # Update the previous step to be gravity
    else:
        ''' General contact ''' # fix here if no gravity is specified
        peelCoh = 'Cohesive_Peel_Int'
        General_Contact(previousStep, peelCoh)
    
    # # Peel Step
    stepName = 'Peel_Test_Dynamic_Explicit'
    descrip = 'Peel the retina away from the vitreous (rotational peel test)'
    peelStepPostGravity(time, stepName, previousStep, descrip, scaleFactor, 
                        MSTI)
    
    updateGeneralContact(stepName, Knn, Kss, Ktt, damageInitiation, 
                          tn, ts, tt, damageEvolution, FE)
    
    if tieInterface == True:
        # Tie the interface together
        VR_sym_tie()
    
    # Boundary Conditions
    Amp()
    if gravity == True:
        peelTestBCUpdate_With_Gravity(stepName)
    else:
        EHR_BC(stepName)
        # GD_BC(stepName) # Not used anymore
        # TD_BC(stepName) # Not used anymore
        
        # Model Outputs
        F_output(stepName)
        H_output(stepName)
    
    Retina_Disp_BC(stepName)
    
    # Undo the spacing to pass in the job description
    global jobDescription
    # replace new lines, spaces, equal signs
    jobDescription = jobDescription.replace('NEWLINE', '\n')
    jobDescription = jobDescription.replace('TAB', '\t')
    jobDescription = jobDescription.replace('SPACE', ' ')
    jobDescription = jobDescription.replace('EQUALSSIGN', '=')
    
    Write_Job(jobName, modelName, jobDescription)
    print('Job has been written')
    Save_INP(jobName)
    Submit_job(jobName)
    print('Job has been submitted')
    del mdb.models['Model-1']

# In[Symmetric Model]
"""
Symmetry
"""

def VR_sym_tie():
    a = abqModel.rootAssembly
    mastSurf=a.instances['R-1'].surfaces['R_V_Surf_BOND']
    slaveSurf=a.instances['V-1'].surfaces['V_R_Surf_BOND']
    abqModel.Tie(name='VR_Tie', 
                 master=mastSurf, 
                 slave=slaveSurf, 
                 positionToleranceMethod=COMPUTED, 
                 adjust=OFF, 
                 tieRotations=ON, 
                 constraintEnforcement=SURFACE_TO_SURFACE, 
                 thickness=ON)


def E_sym_Constrained():
    p = abqModel.parts['E']
    c = p.cells
    pickedCells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    v1, e1, d1 = p.vertices, p.edges, p.datums
    p.PartitionCellByPlaneThreePoints(cells=pickedCells, 
                                      point1=p.InterestingPoint(edge=e1[4], 
                                                                rule=MIDDLE), 
                                      point2=p.InterestingPoint(edge=e1[18], 
                                                                rule=MIDDLE), 
                                      point3=p.InterestingPoint(edge=e1[7], 
                                                                rule=MIDDLE))
    f = p.faces
    p.RemoveFaces(faceList = f[3:4]+f[5:6]+f[7:8]+f[9:12]+f[15:16]+f[17:20]+
                  f[21:22]+f[25:26]+f[27:28]+f[29:30], deleteCells=False)
    
    # Reference point
    p.ReferencePoint(point=(0.0, 0.0, 0.0))
    
    r = p.referencePoints
    refPoints=(r[4], )
    p.Set(referencePoints=refPoints, name='E_RP_Set')
    
    # Sets
    # Edge seeds
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#ffd03fd0 #131f ]', ), )
    p.Set(edges=edges, name='E_Outside_Edge_Seed_Set')
    edges = e.getSequenceFromMask(mask=('[#bc007 #c80 ]', ), )
    p.Set(edges=edges, name='E_Edge_Seed_Set')
    
    # # Define Surface
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#1ffff ]', ), )
    p.Surface(side1Faces=side1Faces, name='E_Surf')
    
    # Remove cells
    c = p.cells
    p.RemoveCells(cellList = c[0:1])
    
    # redefine the E set no w that the cells have been removed
    r = p.referencePoints
    refPoints=(r[4], )
    p.Set(referencePoints=refPoints, name='E_Set')


def G_sym_Constrained():
    
    p = abqModel.parts['G']
    c = p.cells
    pickedCells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    v, e, d = p.vertices, p.edges, p.datums
    p.PartitionCellByPlaneThreePoints(cells=pickedCells, 
                                      point1=p.InterestingPoint(edge=e[10], 
                                                                rule=MIDDLE), 
                                      point2=p.InterestingPoint(edge=e[11], 
                                                                rule=MIDDLE), 
                                      point3=p.InterestingPoint(edge=e[1], 
                                                                rule=MIDDLE))
    
    f1 = p.faces
    p.RemoveFaces(faceList = f1[2:3]+f1[4:5]+f1[7:8]+f1[9:11], 
                  deleteCells=False)
    
    v1, e1, d1, n = p.vertices, p.edges, p.datums, p.nodes
    p.ReferencePoint(point=v1[2])
    
    r = p.referencePoints
    refPoints=(r[4], )
    p.Set(referencePoints=refPoints, name='G_RP_Set')
    
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#2 ]', ), )
    p.Set(faces=faces, name='G_T_Set')
    
    faces = f.getSequenceFromMask(mask=('[#8 ]', ), )
    p.Set(faces=faces, name='G_R_Set')
    
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#3f ]', ), )
    p.Surface(side1Faces=side1Faces, name='G_Surf')
    
    side1Faces = s.getSequenceFromMask(mask=('[#8 ]', ), )
    p.Surface(side1Faces=side1Faces, name='G_R_Surf')
    
    side1Faces = s.getSequenceFromMask(mask=('[#2 ]', ), )
    p.Surface(side1Faces=side1Faces, name='G_T_Surf')
    
    # Remove cells
    c1 = p.cells
    p.RemoveCells(cellList = c1[0:1])
    
    # redefine the set to be the reference point
    r = p.referencePoints
    refPoints=(r[4], )
    p.Set(referencePoints=refPoints, name='G_Set')


def T_sym_constrained():
    p = abqModel.parts['T']
    c = p.cells
    pickedCells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    v, e, d = p.vertices, p.edges, p.datums
    p.PartitionCellByPlaneThreePoints(cells=pickedCells, 
                                      point1=p.InterestingPoint(edge=e[11], 
                                                                rule=MIDDLE), 
                                      point2=p.InterestingPoint(edge=e[7], 
                                                                rule=MIDDLE), 
                                      point3=p.InterestingPoint(edge=e[5], 
                                                                rule=MIDDLE))
    
    f = p.faces
    p.RemoveFaces(faceList = f[1:2]+f[4:5]+f[6:9]+f[11:13], deleteCells=False)
    
    # reference point
    v1, e1, d1, n1 = p.vertices, p.edges, p.datums, p.nodes
    p.ReferencePoint(point=v1[3])
    
    # Sets
    r = p.referencePoints
    refPoints=(r[4], )
    p.Set(referencePoints=refPoints, name='T_RP_Set')
    
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#2 ]', ), )
    p.Set(faces=faces, name='T_G_Set')
    
    # Surfaces
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#ff ]', ), )
    p.Surface(side1Faces=side1Faces, name='T_Surf')
    
    side1Faces = s.getSequenceFromMask(mask=('[#2 ]', ), )
    p.Surface(side1Faces=side1Faces, name='T_G_Surf')
    
    c = p.cells
    p.RemoveCells(cellList = c[0:1])
    
    # Redefine the set to be the reference point
    r = p.referencePoints
    refPoints=(r[4], )
    p.Set(referencePoints=refPoints, name='T_Set')


def R_sym_constrained():
    
    p = abqModel.parts['R']
    c = p.cells
    pickedCells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    v1, e1, d1 = p.vertices, p.edges, p.datums
    p.PartitionCellByPlaneThreePoints(cells=pickedCells, 
                                      point1=p.InterestingPoint(edge=e1[1], 
                                                                rule=MIDDLE), 
                                      point2=p.InterestingPoint(edge=e1[6], 
                                                               rule=MIDDLE), 
                                      point3=p.InterestingPoint(edge=e1[7], 
                                                                rule=MIDDLE))
    
    f1 = p.faces
    p.RemoveFaces(faceList = f1[1:2]+f1[4:5]+f1[6:9]+f1[11:13], 
                  deleteCells=False)
    
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    p.Set(cells=cells, name='R_Set')
    
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#6 ]', ), )
    p.Set(faces=faces, name='R_G_Set')
    
    faces = f.getSequenceFromMask(mask=('[#10 ]', ), )
    p.Set(faces=faces, name='R_V_Set')
    
    faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
    p.Set(faces=faces, name='R_SYM_BC_SET')
    
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#ff ]', ), )
    p.Surface(side1Faces=side1Faces, name='R_Surf')
    
    side1Faces = s.getSequenceFromMask(mask=('[#6 ]', ), )
    p.Surface(side1Faces=side1Faces, name='R_G_Surf')
    
    side1Faces = s.getSequenceFromMask(mask=('[#10 ]', ), )
    p.Surface(side1Faces=side1Faces, name='R_V_Surf_BOND')
    
    side1Faces = s.getSequenceFromMask(mask=('[#1 ]', ), )
    p.Surface(side1Faces=side1Faces, name='R_SYM_BC_SURF')
    
    # Assign section
    region = p.sets['R_Set']
    p.SectionAssignment(region=region, sectionName='Retina_Section', 
                        offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', 
                        thicknessAssignment=FROM_SECTION)


def V_partition_Sphere():
    p = abqModel.parts['V']
    c = p.cells
    pickedCells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    v1, e1, d1 = p.vertices, p.edges, p.datums
    p.PartitionCellByPlaneThreePoints(point1=v1[1], 
                                      point2=v1[0], 
                                      point3=v1[3], 
                                      cells=pickedCells)

    pickedCells = c.getSequenceFromMask(mask=('[#3 ]', ), )
    v2, e, d2 = p.vertices, p.edges, p.datums
    p.PartitionCellByPlaneThreePoints(point1=v2[4], 
                                      point2=v2[1], 
                                      point3=v2[5], 
                                      cells=pickedCells)

    pickedCells = c.getSequenceFromMask(mask=('[#f ]', ), )
    v1, e1, d1 = p.vertices, p.edges, p.datums
    p.PartitionCellByPlaneThreePoints(point1=v1[2], 
                                      point2=v1[5], 
                                      point3=v1[3], 
                                      cells=pickedCells)


def Assembly_sym_constrain():
    a1 = abqModel.rootAssembly
    a1.DatumCsysByDefault(CARTESIAN)
    p = abqModel.parts['E']
    a1.Instance(name='E-1', part=p, dependent=ON)
    p = abqModel.parts['G']
    a1.Instance(name='G-1', part=p, dependent=ON)
    p = abqModel.parts['R']
    a1.Instance(name='R-1', part=p, dependent=ON)
    p = abqModel.parts['T']
    a1.Instance(name='T-1', part=p, dependent=ON)
    p = abqModel.parts['V']
    a1.Instance(name='V-1', part=p, dependent=ON)
    p = abqModel.parts['V_internal']
    a1.Instance(name='V_internal-1', part=p, dependent=ON)


def mergeV_sym():
    # Merge the vitreous and the internal sphere
    a = abqModel.rootAssembly
    a.InstanceFromBooleanMerge(name='V_Merge', 
                               instances=(a.instances['V-1'], 
                                          a.instances['V_internal-1'], ), 
                               keepIntersections=ON, 
                               originalInstances=DELETE, 
                               domain=GEOMETRY)
    
    # Clean up file names after merge
    del abqModel.parts['V']
    del abqModel.parts['V_internal']
    
    abqModel.parts.changeKey(fromName='V_Merge', toName='V')
    a = abqModel.rootAssembly
    a.regenerate()
    abqModel.rootAssembly.features.changeKey(fromName='V_Merge-1', 
                                             toName='V-1')
    
    p = abqModel.parts['V']
    f = p.faces
    p.RemoveFaces(faceList = f[0:3]+f[4:5]+f[8:9]+f[12:13]+f[15:16]+
                  f[19:21]+f[23:24]+f[26:27]+f[28:29]+f[32:36], 
                  deleteCells=False)
    
    a.regenerate()


def V_sym_constrained():
    # Partition V along the width of the retina
    p = abqModel.parts['V']
    p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=-0.00254)
    abqModel.parts['V'].features.changeKey(fromName='Datum plane-1', 
                                           toName='Retina_Width')
    c = p.cells
    pickedCells = c.getSequenceFromMask(mask=('[#14 ]', ), )
    d1 = p.datums
    p.PartitionCellByDatumPlane(datumPlane=d1[3], cells=pickedCells)
    
    # # Left side of vitreous 22.5 degrees
    # # Right side of vitreous 18.875000
    p.DatumAxisByPrincipalAxis(principalAxis=ZAXIS)
    p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=0.0)
    
    d2 = p.datums
    p.DatumPlaneByRotation(plane=d2[6], axis=d2[5], angle=18.75)
    
    d1 = p.datums
    ''' angle=-22.5 was the previous angle for the back side of the retina, 
    because of extreme element deformation, a new model was Created'''
    # ang = -22.5 # Previous
    ang = -18.75*2 # Updated angle
    p.DatumPlaneByRotation(plane=d2[7], axis=d1[5], angle=ang)
    c = p.cells
    pickedCells = c.getSequenceFromMask(mask=('[#2 ]', ), )
    d2 = p.datums
    p.PartitionCellByDatumPlane(datumPlane=d2[7], cells=pickedCells)
    
    pickedCells = c.getSequenceFromMask(mask=('[#40 ]', ), )
    d1 = p.datums
    p.PartitionCellByDatumPlane(datumPlane=d1[8], cells=pickedCells)
    
    # Define sets
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#fff ]', ), )
    p.Set(cells=cells, name='V_Set')
    
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#8080 ]', ), )
    p.Set(faces=faces, name='V_R_Set')
    
    # Symmetric BC
    faces = f.getSequenceFromMask(mask=('[#17000042 #6a ]', ), )
    p.Set(faces=faces, name='V_SYM_BC_SET')
    
    # Surfaces
    s = p.faces
    side1Faces = s.getSequenceFromMask(mask=('[#1700a0ca #7ea ]', ), )
    p.Surface(side1Faces=side1Faces, name='V_Surf')
    side1Faces = s.getSequenceFromMask(mask=('[#8080 ]', ), )
    p.Surface(side1Faces=side1Faces, name='V_R_Surf_BOND')
    
    # Symmetric BC
    side1Faces = s.getSequenceFromMask(mask=('[#17000042 #6a ]', ), )
    p.Surface(side1Faces=side1Faces, name='V_SYM_BC_SURF')
    
    # Assign section
    region = p.sets['V_Set']
    p.SectionAssignment(region=region, sectionName='Vitreous_Section', 
                        offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', 
                        thicknessAssignment=FROM_SECTION)


def E_sym_constrain_msh(e1Seed, e2Seed):
    p = abqModel.parts['E']
    e = p.edges
    pickedEdges = e.getSequenceFromMask(mask=('[#bc007 #c80 ]', ), )
    p.seedEdgeBySize(edges=pickedEdges, 
                     size=e1Seed, 
                     deviationFactor=0.1, 
                     constraint=FINER)
    pickedEdges = e.getSequenceFromMask(mask=('[#ffd03fd0 #131f ]', ), )
    p.seedEdgeBySize(edges=pickedEdges, 
                     size=e2Seed, 
                     deviationFactor=0.1, 
                     constraint=FINER)
    elemType1 = mesh.ElemType(elemCode=R3D4, elemLibrary=EXPLICIT)
    elemType2 = mesh.ElemType(elemCode=R3D3, elemLibrary=EXPLICIT)
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#1ffff ]', ), )
    pickedRegions =(faces, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
    # (unique node numbering)
    p.setValues(startNodeLabel=1000000, startElemLabel=1000000)
    p.generateMesh()


def G_sym_constrain_msh(gSeed):
    p = abqModel.parts['G']
    p.seedPart(size=gSeed, deviationFactor=0.1, minSizeFactor=0.1)
    elemType1 = mesh.ElemType(elemCode=R3D4, elemLibrary=EXPLICIT)
    elemType2 = mesh.ElemType(elemCode=R3D3, elemLibrary=EXPLICIT)
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#3f ]', ), )
    pickedRegions =(faces, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
    # (unique node numbering)
    p.setValues(startNodeLabel=2000000, startElemLabel=2000000)
    p.generateMesh()


def T_sym_constrain_msh(ptSeed):
    p = abqModel.parts['T']
    p.seedPart(size=ptSeed, deviationFactor=0.1, minSizeFactor=0.1)
    elemType1 = mesh.ElemType(elemCode=R3D4, elemLibrary=EXPLICIT)
    elemType2 = mesh.ElemType(elemCode=R3D3, elemLibrary=EXPLICIT)
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#ff ]', ), )
    pickedRegions =(faces, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
    # (unique node numbering)
    p.setValues(startNodeLabel=3000000, startElemLabel=3000000)
    p.generateMesh()


def R_sym_constrain_msh(rSeed):
    p = abqModel.parts['R']
    e = p.edges
    pickedEdges = e.getSequenceFromMask(mask=('[#3ffff ]', ), )
    p.seedEdgeBySize(edges=pickedEdges, 
                     size=rSeed, 
                     deviationFactor=0.1, 
                     constraint=FINER)
    c = p.cells
    pickedRegions = c.getSequenceFromMask(mask=('[#1 ]', ), )
    p.setMeshControls(regions=pickedRegions, technique=SWEEP, 
        algorithm=ADVANCING_FRONT)
    c, e1 = p.cells, p.edges
    p.setSweepPath(region=c[0], edge=e1[3], sense=REVERSE)
    elemType1 = mesh.ElemType(elemCode=C3D8R, 
                              elemLibrary=EXPLICIT, 
                              kinematicSplit=AVERAGE_STRAIN, 
                              secondOrderAccuracy=ON, 
                              hourglassControl=ENHANCED, 
                              distortionControl=ON, 
                              lengthRatio=0.100000001490116)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=EXPLICIT)
    elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=EXPLICIT)
    # c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    pickedRegions =(cells, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, 
                                                       elemType2, 
                                                       elemType3))
    # (unique node numbering)
    p.setValues(startNodeLabel=4000000, startElemLabel=4000000)
    p.generateMesh()


def VseedPart(v2Seed):
    ''' Seed the entire vitreous '''
    p = abqModel.parts['V']
    p.seedPart(size=v2Seed, deviationFactor=0.1, minSizeFactor=0.1)


def V_SeedTop(v1Seed):
    ''' Seed the top of the vitreous where the retina is bonded '''
    p = abqModel.parts['V']
    e = p.edges
    pickedEdges = e.getSequenceFromMask(mask=('[#ffffffff #f ]', ), )
    p.seedEdgeBySize(edges=pickedEdges, 
                     size=v1Seed, 
                     deviationFactor=0.1, 
                     constraint=FINER)


def vitreous_seed_bias(v1Seed, v2Seed):
    ''' Seed the outside edges of the vitreous leading up to the bonded 
    interface with biased mesh to weight the attachment area '''
    p = abqModel.parts['V']
    e = p.edges
    pickedEdges1 = e.getSequenceFromMask(mask=('[#0 #100040 ]', ), )
    pickedEdges2 = e.getSequenceFromMask(mask=('[#0 #400000 ]', ), )
    p.seedEdgeByBias(biasMethod=SINGLE, 
                     end1Edges=pickedEdges1, 
                     end2Edges=pickedEdges2, 
                     minSize=v1Seed, 
                     maxSize=v2Seed, 
                     constraint=FINER)


def vitreous_Seed_Bottom_Bias(v1Seed, v2Seed):
    ''' Seed the bottom of the vitreous '''
    p = abqModel.parts['V']
    e = p.edges
    pickedEdges1 = e.getSequenceFromMask(mask=('[#0 #10000 ]', ), )
    pickedEdges2 = e.getSequenceFromMask(mask=('[#0 #1000 ]', ), )
    p.seedEdgeByBias(biasMethod=SINGLE, 
                     end1Edges=pickedEdges1, 
                     end2Edges=pickedEdges2, 
                     minSize=v1Seed, 
                     maxSize=v2Seed, 
                     constraint=FINER)


def vHex():
    ''' Hexahedral mesh definition for the vitreous '''
    p = abqModel.parts['V']
    elemType1 = mesh.ElemType(elemCode=C3D8R, 
                              elemLibrary=EXPLICIT, 
                              kinematicSplit=AVERAGE_STRAIN, 
                              secondOrderAccuracy=ON, 
                              hourglassControl=ENHANCED, 
                              distortionControl=ON, 
                              lengthRatio=0.100000001490116)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=EXPLICIT)
    elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=EXPLICIT)
    
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#4bf ]', ), )
    pickedRegions =(cells, )
    p.setElementType(regions=pickedRegions, 
                     elemTypes=(elemType1, elemType2, elemType3))


def vTet():
    ''' Tetrahedral mesh definition for the vitreous '''
    p = abqModel.parts['V']
    c = p.cells
    pickedRegions = c.getSequenceFromMask(mask=('[#b40 ]', ), )
    p.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE)
    elemType1 = mesh.ElemType(elemCode=C3D20R)
    elemType2 = mesh.ElemType(elemCode=C3D15)
    elemType3 = mesh.ElemType(elemCode=C3D10)
    
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#b40 ]', ), )
    pickedRegions =(cells, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
        elemType3))
    elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=EXPLICIT)
    elemType3 = mesh.ElemType(elemCode=C3D4, 
                              elemLibrary=EXPLICIT, 
                              secondOrderAccuracy=ON, 
                              distortionControl=ON, 
                              lengthRatio=0.100000001490116)
    
    cells = c.getSequenceFromMask(mask=('[#b40 ]', ), )
    pickedRegions =(cells, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
        elemType3))


def V_generate_mesh():
    ''' Mesh the vitreous '''
    p = abqModel.parts['V']
    # (unique node numbering)
    p.setValues(startNodeLabel=5000000, startElemLabel=5000000)
    p.generateMesh()


def V_sym_constrain_msh(v1Seed, v2Seed):
    ''' Mesh the vitreous with the two different seed sizes
    Seed the part
    Seed the top
    bias the edge
    seed the bottom
    hexahedral elements
    tetrahedral elements
    generate mesh '''
    VseedPart(v2Seed)
    V_SeedTop(v1Seed)
    vitreous_seed_bias(v1Seed, v2Seed)
    vitreous_Seed_Bottom_Bias(v1Seed, v2Seed)
    vHex()
    vTet()
    V_generate_mesh()



def V_SYM_Constrain_BC(stepName):
    a = abqModel.rootAssembly
    f = a.instances['V-1'].faces
    faces = f.getSequenceFromMask(mask=('[#17000042 #6a ]', ), )
    region = a.Set(faces=faces, name='V_SYM_BC_SET')
    abqModel.ZsymmBC(name='V_sym', 
                     createStepName=stepName, 
                     region=region, 
                     localCsys=None)


def R_SYM_Constrain_BC(stepName):
    a = abqModel.rootAssembly
    f = a.instances['R-1'].faces
    faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
    region = a.Set(faces=faces, name='R_SYM_BC_SET')
    abqModel.ZsymmBC(name='R_sym', 
                     createStepName=stepName, 
                     region=region, 
                     localCsys=None)


""" Write the FEA Code """
def FEA_Symmetry():
    """
    Function that generates FEA code to model vitreoretinal adhesion
    
    # Steps are as follows:
        1 - Create new model database
        2 - Import SolidWorks STEP file (Includes all parts)
        3 - Material property definitions
        4 - Part features (Element & Node Sets & Reference Points ...)
        5 - Mesh parts (Specify seed size)
        6 - Assembly
        7 - Step (Dynamic Explicit with Mass Scaling)
        8 - Outputs (Field & History)
        9 - Contact (General Contact)
        10 - Contact pair (Retina/Vitreous - Bonded Surface)
        11 - Tie Constraint (Retina - Glue)
        12 - Amplitude definition
        13 - BC's'
        14 - Submit Job  :)
    """
    
    # Import SolidWorks STEP file
    ImportStepEyeConstrained()
    
    # Mat Props
    Retina_Mat_Prop(RetinaProp)
    Vitreous_Mat_Prop(VitreousProp)
    
    """ Constrained vitreous """
    # Pat Geometry/RPs/Sets/Surfaces
    E_sym_Constrained()
    G_sym_Constrained()
    T_sym_constrained()
    R_sym_constrained()
    
    # Define and then merge in the assembly to reduce computational time
    V_Internal_Sphere()
    
    # # Assembly
    Assembly_sym_constrain()
    
    # partition Vitreous x,y,z plane
    V_partition_Sphere()
    
    # Merge V and V Int
    mergeV_sym()
    
    # Update V sets
    V_sym_constrained()
    
    # Mesh parts
    E_sym_constrain_msh(e1Seed, e2Seed)
    G_sym_constrain_msh(gSeed)
    T_sym_constrain_msh(ptSeed)
    V_sym_constrain_msh(v1Seed, v2Seed)
    R_sym_constrain_msh(rSeed)
    
    # # Convert Hexahedral elements to quadratic tetrahedral elements
    # QuadraticTetVitreous()
    # QuadraticTetRetina()
    
    # Eliminate the glue and tab from the model
    a = abqModel.rootAssembly
    a.features['G-1'].suppress()
    a.features['T-1'].suppress()
    
    # Gravity Step
    previousStep = 'Initial'
    if gravity == True:
        stepName = 'Gravity_Step'
        descrip = ('Applying gravity to the model and letting the ' + 
                   'vitreous and retina settle')
        
        GravityStep(200, previousStep, scaleFactor, 0.03125, stepName, descrip)
        Gravity(stepName)
        smoothGravity()
        
        # Interactions
        cohTieName = 'Cohesive_Gravity_Tie'
        General_Contact(stepName, cohTieName)
        
        # Interaction properties
        turnTieCohesive(stepName, cohTieName)
        
        V_SYM_Constrain_BC(stepName)
        R_SYM_Constrain_BC(stepName)
        
        # Zero movement boundary conditions
        Amp()
        EHR_BC_Fixed(stepName)
        
        # # Model outputs for gravity step
        F_output(stepName)
        H_output(stepName)
        
        previousStep = stepName # Update the previous step to be gravity
    else:
        ''' General contact ''' # fix here if no gravity is specified
        peelCoh = 'Cohesive_Peel_Int'
        General_Contact(previousStep, peelCoh)
    
    # # Peel Step
    stepName = 'Peel_Test_Dynamic_Explicit'
    descrip = 'Peel the retina away from the vitreous (rotational peel test)'
    peelStepPostGravity(time, stepName, previousStep, descrip, scaleFactor, 
                        MSTI)
    
    updateGeneralContact(stepName, Knn, Kss, Ktt, damageInitiation, 
                          tn, ts, tt, damageEvolution, FE)
    
    if tieInterface == True:
        # Tie the interface together
        VR_sym_tie()
    
    # Boundary Conditions
    Amp()
    if gravity == True:
        peelTestBCUpdate_With_Gravity(stepName)
    
    else:
        EHR_BC(stepName)
        # GD_BC(stepName) # Not used anymore
        # TD_BC(stepName) # Not used anymore
        V_SYM_Constrain_BC(stepName)
        R_SYM_Constrain_BC(stepName)
        
        # Model Outputs
        F_output(stepName)
        H_output(stepName)
    
    Retina_Disp_BC(stepName)
    
    # Undo the spacing to pass in the job description
    global jobDescription
    # replace new lines, spaces, equal signs
    jobDescription = jobDescription.replace('NEWLINE', '\n')
    jobDescription = jobDescription.replace('TAB', '\t')
    jobDescription = jobDescription.replace('SPACE', ' ')
    jobDescription = jobDescription.replace('EQUALSSIGN', '=')
    
    Write_Job(jobName, modelName, jobDescription)
    print('Job has been written')
    Save_INP(jobName)
    Submit_job(jobName)
    print('Job has been submitted')
    del mdb.models['Model-1']


# In[Main import info]

if __name__ == '__main__':
    """ Run the following function """
    
    # Print File of tests & attributes ran in order to make sure they are 
    # being properly pass through
    print("\nWriting out the Argument Data...")
    filename = os.path.join(abqWD, 'FEAArgumentData' + '.txt')
    outfile = open(filename,'w')
    outfile.write('sys.argv\n')
    outfile.write('\n'.join(sys.argv)) # write all arguments passed into abaqus
    outfile.close()
    print("\nDone!")
    print("\nThe output file will be named '{}".format(filename) + "'")
    print("\nIt will be in the same working directory as your Abaqus model\n")
    
    # # Testing when importing into abaqus script
    # gravity =              eval('True') # gravity
    # symmetry =             eval('False') # symmetry
    # simplified=            eval('True') # simplified model
    # modelName =                 'T1Si' # model name 
    # jobName =                   'test' # file name/job name
    # time =                float('100')
    # e1Seed =                    '[10,1,0.0009765625]'
    # e2Seed =                    '[8,1,0.00390625]'
    # ptSeed =                    '[6,1,0.015625]'
    # gSeed =                     '[7,1,0.0078125]'
    # v1Seed =                    '[10,1,0.0009765625]' # '[11.38,1,0.00037521366730664343]' # 
    # v2Seed =                    '[8,1,0.00390625]'
    # rSeed =                     '[10,1,0.0009765625]' # '[11.3275,1,0.0003891192571059363]' # 
    # scaleFactor =               '[0,1,1]'
    # MSTI =                      '[4,1,0.0625]' # MassScaleTimeIncrement
    # RetinaProp =          float('11120.0') # Young's modulus for retina
    # VitreousProp =        float('100') # '69.56549028991259') # Young's modulus for vitreous 386.717932801091
    # KnnString =              str([26.21216496521396,1,77740603.15760481]) #
    # KssString =              str([27.992885300905385,1,267114916.34363237]) #
    # KttString =              str([27.65583405906571,1,211463592.90645516])
    # damageInitiation =          True # True/False
    # tnString =               str([18.830816653206917,1,466273.4160693089])
    # tsString =               str([17.49221225177773,1,184365.8917311695]) # Damage initiation
    # ttString =               str([6.5328659715983814,1,92.59523006880973]) # Damage initiation
    # damageEvolution =           True # True/false convert to bool
    # FEString =               str([-0.9185766704351879,1,0.5290306923507394])
    # OptimizationStatus =        True
    # tieInterface =              True
    # jobDescription =            """Test""" #'Test MODEL Cube Script'
    
    
    # Pass in arguments from previous file Strip the brackets from the strings
    gravity =              eval(sys.argv[-29]) # gravity
    symmetry =             eval(sys.argv[-28]) # symmetry
    simplified =           eval(sys.argv[-27]) # simplified model
    modelName =                 sys.argv[-26] # model name
    jobName =                   sys.argv[-25] # file name/job name
    time =                float(sys.argv[-24])
    e1Seed =                    sys.argv[-23]
    e2Seed =                    sys.argv[-22]
    ptSeed =                    sys.argv[-21]
    gSeed =                     sys.argv[-20]
    v1Seed =                    sys.argv[-19]
    v2Seed =                    sys.argv[-18]
    rSeed =                     sys.argv[-17]
    scaleFactor =               sys.argv[-16]
    MSTI =                      sys.argv[-15] # MassScaleTimeIncrement
    RetinaProp =          float(sys.argv[-14]) # Young's modulus for retina
    VitreousProp =        float(sys.argv[-13]) # Young's modulus for vitreous
    KnnString =                 sys.argv[-12] # Cohesive behavior
    KssString =                 sys.argv[-11] # Cohesive behavior
    KttString =                 sys.argv[-10] # Cohesive behavior
    damageInitiation =     eval(sys.argv[-9]) # True/false convert to bool
    tnString =                  sys.argv[-8] # Damage initiation
    tsString =                  sys.argv[-7] # Damage initiation
    ttString =                  sys.argv[-6] # Damage initiation
    damageEvolution =      eval(sys.argv[-5]) # True/false convert to bool
    FEString =                  sys.argv[-4] # Fracture energy
    optimizationStatus =        sys.argv[-3] # None/variables to be optimized
    tieInterface =         eval(sys.argv[-2]) # True/false convert to bool
    jobDescription =            sys.argv[-1] # String
    
    
    """ Convert the strings back to lists of floats """
    e1SeedStrip = str(e1Seed)[1:-1] # Strip the brackets from the string
    e1SeedList = [float(i) for i in e1SeedStrip.split(',')]
    e1Seed = e1SeedList[2] # value
    
    e2SeedStrip = str(e2Seed)[1:-1] # Strip the brackets from the string
    e2SeedList = [float(i) for i in e2SeedStrip.split(',')]
    e2Seed = e2SeedList[2] # value
    
    ptSeedStrip = str(ptSeed)[1:-1] # Strip the brackets from the string
    ptSeedList = [float(i) for i in ptSeedStrip.split(',')]
    ptSeed = ptSeedList[2] # value
    
    gSeedStrip = str(gSeed)[1:-1] # Strip the brackets from the string
    gSeedList = [float(i) for i in gSeedStrip.split(',')]
    gSeed = gSeedList[2] # value
    
    v1SeedStrip = str(v1Seed)[1:-1] # Strip the brackets from the string
    v1SeedList = [float(i) for i in v1SeedStrip.split(',')]
    v1Seed = v1SeedList[2] # value
    
    v2SeedStrip = str(v2Seed)[1:-1] # Strip the brackets from the string
    v2SeedList = [float(i) for i in v2SeedStrip.split(',')]
    v2Seed = v2SeedList[2] # value
    
    rSeedStrip = str(rSeed)[1:-1] # Strip the brackets from the string
    rSeedList = [float(i) for i in rSeedStrip.split(',')]
    rSeed = rSeedList[2] # value
    
    # Strip the brackets from the string
    scaleFactorStrip = str(scaleFactor)[1:-1] 
    scaleFactorList = [float(i) for i in scaleFactorStrip.split(',')]
    scaleFactor = scaleFactorList[2] # value
    
    # Strip the brackets from the string
    # MassScaleTimeIncrement
    MSTIStrip = str(MSTI)[1:-1]
    MSTIList = [float(i) for i in MSTIStrip.split(',')]
    MSTI = MSTIList[2] # value
    
    KnnStrip = str(KnnString)[1:-1] # Strip the brackets from the string
    KnnList = [float(i) for i in KnnStrip.split(',')]
    Knn = KnnList[2] # value
    
    KssStrip = str(KssString)[1:-1] # Strip the brackets from the string
    KssList = [float(i) for i in KssStrip.split(',')]
    Kss = KssList[2] # value
    
    KttStrip = str(KttString)[1:-1] # Strip the brackets from the string
    KttList = [float(i) for i in KttStrip.split(',')]
    Ktt = KttList[2] # value
    
    tnStrip = str(tnString)[1:-1] # Strip the brackets from the string
    tnList = [float(i) for i in tnStrip.split(',')]
    tn = tnList[2] # value
    
    tsStrip = str(tsString)[1:-1] # Strip the brackets from the string
    tsList = [float(i) for i in tsStrip.split(',')]
    ts = tsList[2] # value
    
    ttStrip = str(ttString)[1:-1] # Strip the brackets from the string
    ttList = [float(i) for i in ttStrip.split(',')]
    tt = ttList[2] # value
    
    FEStrip = str(FEString)[1:-1] # Strip the brackets from the string
    FEList = [float(i) for i in FEStrip.split(',')]
    FE = FEList[2] # value
    
    """ Write the FEA Code """
    Mdb()
    modelDescription = ('Measure adhesion between the retina & vitreous of ' + 
                        'the human eye')
    abqModel = mdb.Model(name=modelName, 
                         description=modelDescription, 
                         modelType=STANDARD_EXPLICIT, 
                         copyInteractions=ON, 
                         copyConstraints=ON)
    
    if symmetry == True:
        print('FEA SYM model')
        FEA_Symmetry()
    else:
        print('FEA Non-SYM model')
        FEA()
