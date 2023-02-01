"""
Created on Wed Jun 17 16:48:49 2020

@author: Kiffer Creveling
Instructions:
    1) Save this script in a folder containing your ODB file
    2) Open a command window and navigate to your directory containing this script and your ODB file
    3) Create a .bat file
    3) Issue the command to call the script and extract data:
        abaqus python -c "import BpT; BpT.data_extract('xxxxxxx.odb')"
"""
# **************************
from odbAccess import *
import odbAccess as oa
from sys import argv, exit
from abaqusConstants import *
from textRepr import *
import pdb
import numpy as np
import os

""" Pass arguments into this script """
script =             sys.argv[0]
jobName =            sys.argv[1]
gravity =       eval(sys.argv[2]) # True/False
symmetry =      eval(sys.argv[3]) # True/False
simplified =    eval(sys.argv[4]) # True/False
BondStatus =    eval(sys.argv[5]) # True/False
PDFMStatus =    eval(sys.argv[6]) # True/False # not used in the extraction

def openOdb(jobName):
    """
    Function used to locate the .odb given a file name
    
    Parameters
    ----------
    jobName : Name of the ABAQUS .odb file
    
    Returns
    -------
    odb : Abaqus output file
    """
    if jobName.endswith('.odb'):
        odbFile = jobName
        try:
            odb=oa.openOdb(path=odbFile, readOnly=TRUE)
            print("\nOpening the odb file... (.odb was specified)")
            return odb
        except:
            print("ERROR: Unable to open the specified odb %s.  Exiting." 
                  % odbFile)
            exit(0)
    
    else:
        odbFile = jobName + '.odb'
        # Try opening the odb file
        try:
            odb=oa.openOdb(path=odbFile, readOnly=TRUE)
            print("\nOpening the odb file... (Searching for .odb)")
            return odb
        except:
            print("ERROR: Unable to open the specified odb %s.  Exiting." 
                  % odbFile)
            exit(0)

def data_extract(jobName):
    """
    Function used to extract data from the .odb file
    
    Parameters
    ----------
    jobName : The name of ABAQUS .odb file
    
    Returns
    -------
    Two files of data used for plotting
    """
    
    # due to symmetry multiply the values by 2
    if symmetry == True:
        mult = 2
    else:
        mult = 1
    
    frames = []
    try:
        odb = openOdb(jobName)
    except:
        print(os.getcwd())
        print("Looks like there is a problem with the job name or odb file")
    
    theta = 30
    LoadCellDirection = [np.cos(theta*np.pi/180), np.sin(theta*np.pi/180), 0]
    
    """ Field Output data arrays """
    RF = []
    
    # vector components of the reaction force
    RFx = []
    RFy = []
    RFz = []
    
    U_top = [] # values to append
    U_bot = [] # values to append
    Nforc = []
    
    # Used to calculate bond distance
    R_bot = [] # bottom of retina
    V_top = [] # top of vitreous
    Bond_disp = [] # Bond separation distance
    
    CnormF_RV = []
    CnormF_VR = []
    Cpress_RV = []
    Cpress_VR = []
    Cshear1_RV = []
    Cshear1_VR = []
    Cshear2_RV = []
    Cshear2_VR = []
    CshearF_RV = []
    CshearF_VR = []
    
    # Cpress_RV = []
    # Cpress_VR = []
    Cpress_RV_AVG = []
    Cpress_VR_AVG = []
    frames = [] # List of frames
    time = [] # Time array
    
    # Used for reaction force simplicity further in the code
    temp = [] # Temporary array used for iterating (Clears after each iteration)
    tempx = []
    tempy = []
    tempz = []
    
    # List variables for exporting data
    CnormF_RV_List = []
    CnormF_VR_List = []
    Cpress_RV_List = []
    Cpress_VR_List = []
    Cshear1_RV_List = []
    Cshear1_VR_List = []
    Cshear2_RV_List = []
    Cshear2_VR_List = []
    CshearF_RV_List = []
    CshearF_VR_List = []
    
    """ History Output data arrays """
    Hist_Time = []
    IE = []
    KE = []
    CAreaCP_RG = []
    CAreaCP_GR = []
    CAreaCP_RV = []
    CAreaCP_VR = []
    CFNCP_RG = []
    CFNCP_GR = []
    CFNCP_RV = []
    CFNCP_VR = []
    Glue_RP_RF = []
    
    """ Loop over the field outputs"""
    step = odb.steps.keys() # determines the step in the abaqus odb file (typically displacement)
    disp_step = step[0] # Defines the step as a variable name
    for frame, odbFrame in enumerate(odb.steps[disp_step].frames):
        frames.append(frame) # Construct a list of all of the frames
        
        """ Extract ODB fieldOutputs """
        fieldOutput = odbFrame.fieldOutputs
        
        # Print the time during the simulation
        print(odbFrame.description)
        time.append(odbFrame.frameValue)
        
        """ Abaqus Instances (Parts) """
        odbInstance = odb.rootAssembly.instances
        
        if simplified == False:
            # If Simp is not in the title
            part_E = odbInstance.keys(0)[0]
            part_G = odbInstance.keys(0)[1]
            part_R = odbInstance.keys(0)[2]
            part_T = odbInstance.keys(0)[3]
            part_V = odbInstance.keys(0)[4]
            
        elif simplified == True:
            # If simplification exists, omit the glue & tab
            part_E = odbInstance.keys(0)[0]
            part_R = odbInstance.keys(0)[1]
            part_V = odbInstance.keys(0)[2]
        else:
            print('Error in part definitions')
        
        """ Nodal displacements """
        fieldObject_U = fieldOutput['U'] # displacements
        
        if simplified == False:
            # If Simp is not in the title
            
            # Glue
            Displacements = fieldObject_U.getSubset(region=odbInstance[part_G].nodeSets['G_RP_SET'])
            for Uyi in Displacements.values: # Loops over each node in the "SET" defined by the displacement
                Uyi_vec = [Uyi.data[0], Uyi.data[1], Uyi.data[2]]
                # Find the magnitude
                temp.append(np.dot(Uyi_vec, LoadCellDirection)) # Creates a list of displacements in the "SET"
                
            SU = np.sum(temp) # Sums up the list of displacements from the "SET"
            AvgU_top = SU/len(temp) # Divide by the number of nodes in the set to get average
            U_top.append(AvgU_top) # Adds the total displacement to the U-array by summing across each step
            temp = [] # Clear the array for the next iteration in the loop
            
        elif simplified == True:
            # If simplification exists, omit the values
            
            Displacements = fieldObject_U.getSubset(region=odbInstance[part_R].nodeSets['R_G_SET'])
            for Uyi in Displacements.values: # Loops over each node in the "SET" defined by the displacement
                Uyi_vec = [Uyi.data[0], Uyi.data[1], Uyi.data[2]]
                # Find the magnitude
                temp.append(np.dot(Uyi_vec, LoadCellDirection)) # Creates a list of displacements in the "SET"
                
            SU = np.sum(temp) # Sums up the list of displacements from the "SET"
            AvgU_top = SU/len(temp) # Divide by the number of nodes in the set to get average
            U_top.append(AvgU_top) # Adds the total displacement to the U-array by summing across each step
            temp = [] # Clear the array for the next iteration in the loop
            
        else:
            print('Error in nodal displacements')
        
        """ Bond Distance """
        Displacements = fieldObject_U.getSubset(region=odbInstance[part_R].nodeSets['R_V_SET'])
        for Uyi in Displacements.values: # Loops over each node in the "SET" defined by the displacement
            Uyi_vec = [Uyi.data[0], Uyi.data[1], Uyi.data[2]]
            # Find the magnitude
            temp.append(np.dot(Uyi_vec, LoadCellDirection)) # Creates a list of displacements in the "SET"
            
        SU = np.sum(temp) # Sums up the list of displacements from the "SET"
        AvgR_bot = SU/len(temp) # Divide by the number of nodes in the set to get average
        R_bot.append(AvgR_bot) # Adds the total displacement to the U-array by summing across each step
        temp = [] # Clear the array for the next iteration in the loop
            
        Displacements = fieldObject_U.getSubset(region=odbInstance[part_V].nodeSets['V_R_SET'])
        for Uyi in Displacements.values: # Loops over each node in the "SET" defined by the displacement
            Uyi_vec = [Uyi.data[0], Uyi.data[1], Uyi.data[2]]
            # Find the magnitude
            temp.append(np.dot(Uyi_vec, LoadCellDirection)) # Creates a list of displacements in the "SET"
            
        SU = np.sum(temp) # Sums up the list of displacements from the "SET"
        AvgV_top = SU/len(temp) # Divide by the number of nodes in the set to get average
        V_top.append(AvgV_top) # Adds the total displacement to the U-array by summing across each step
        temp = [] # Clear the array for the next iteration in the loop
        
        # average difference in nodal positions between the *bonded surfaces
        Bond_disp.append(AvgR_bot - AvgV_top)
        
        if BondStatus == True:
        # if fieldOutput.has_key('CNORMF   ASSEMBLY_R-1_R_V_SURF_BOND/ASSEMBLY_V-1_V_R_SURF_BOND') == 1: # if the repository has the item
            # 'CNORMF   ASSEMBLY_CP-R_V/ASSEMBLY_CP-V_R' # This was used when the contact pair was being defined by abaqus.  Now that it is defined as a keyword, it is slightly different
            
            # Contact Force
            fieldObject_CNORMF_RV = fieldOutput.keys()[0]
            fieldObject_CNORMF_VR = fieldOutput.keys()[1]
        
            # Contact Stress
            fieldObject_CPRESS_RV = fieldOutput.keys()[2]
            fieldObject_CPRESS_VR = fieldOutput.keys()[3]
            
            # Contact Shear1
            fieldObject_CSHEAR1_RV = fieldOutput.keys()[4]
            fieldObject_CSHEAR1_VR = fieldOutput.keys()[5]
            
            # Contact Shear2
            fieldObject_CSHEAR2_RV = fieldOutput.keys()[6]
            fieldObject_CSHEAR2_VR = fieldOutput.keys()[7]
            
            # Contact ShearF
            fieldObject_CSHEARF_RV = fieldOutput.keys()[8]
            fieldObject_CSHEARF_VR = fieldOutput.keys()[9]
            
            """ Contact Force (Retina-Vitreous) """
            # Retina-Vitreous contact stress
            for CnF_RV_i in fieldOutput[fieldObject_CNORMF_RV].values:
                temp.append(CnF_RV_i.data*mult)
                
            S_CnF_RV = np.sum(temp) # Sums up the list of stress from the "SET"
            CnormF_RV_List.append(temp) # append the list of nodal values
            CnormF_RV.append(S_CnF_RV) # Adds the total stress to the stress-array by summing across each step
            temp = [] # Clear the array for the next iteration in the loop
            
            """ Contact Force (Vitreous-Retina) """
            # Retina-Vitreous contact stress
            for CnF_VR_i in fieldOutput[fieldObject_CNORMF_VR].values:
                temp.append(CnF_RV_i.data*mult)
                
            S_CnF_VR = np.sum(temp) # Sums up the list of stress from the "SET"
            CnormF_VR_List.append(temp) # append the list of nodal values
            CnormF_VR.append(S_CnF_VR) # Adds the total stress to the stress-array by summing across each step
            temp = [] # Clear the array for the next iteration in the loop
            
            """ Contact Stress (Retina-Vitreous) """
            # Retina-Vitreous contact stress
            for CP_RV_i in fieldOutput[fieldObject_CPRESS_RV].values:
                temp.append(CP_RV_i.data*mult)
                
            S_CP_RV = np.sum(temp) # Sums up the list of stress from the "SET"
            Cpress_RV_List.append(temp)
            Cpress_RV.append(S_CP_RV) # Adds the total stress to the stress-array by summing across each step
            Cpress_RV_AVG.append(S_CP_RV/len(fieldOutput[fieldObject_CPRESS_RV].values))
            temp = [] # Clear the array for the next iteration in the loop
            
            """ Contact Stress (Vitreous-Retina) """
            # Retina-Vitreous contact stress
            for CP_VR_i in fieldOutput[fieldObject_CPRESS_VR].values:
                temp.append(CP_VR_i.data*mult)
                
            S_CP_VR = np.sum(temp) # Sums up the list of stress from the "SET"
            Cpress_VR_List.append(temp)
            Cpress_VR.append(S_CP_VR) # Adds the total stress to the stress-array by summing across each step
            Cpress_VR_AVG.append(S_CP_VR/len(fieldOutput[fieldObject_CPRESS_VR].values))
            temp = [] # Clear the array for the next iteration in the loop
            
            """ Contact Shear1 (Retina-Vitreous) """
            # Retina-Vitreous contact stress
            for Cs1_RV_i in fieldOutput[fieldObject_CSHEAR1_RV].values:
                temp.append(Cs1_RV_i.data*mult)
            
            S_Cs1_RV = np.sum(temp) # Sums up the list of stress from the "SET"
            Cshear1_RV_List.append(temp) # append the list of nodal values
            Cshear1_RV.append(S_Cs1_RV) # Adds the total stress to the stress-array by summing across each step
            temp = [] # Clear the array for the next iteration in the loop
            
            """ Contact Shear1 (Vitreous-Retina) """
            # Retina-Vitreous contact stress
            for Cs1_VR_i in fieldOutput[fieldObject_CSHEAR1_VR].values:
                temp.append(Cs1_VR_i.data*mult)
                
            S_Cs1_VR = np.sum(temp) # Sums up the list of stress from the "SET"
            Cshear1_VR_List.append(temp) # append the list of nodal values
            Cshear1_VR.append(S_Cs1_VR) # Adds the total stress to the stress-array by summing across each step
            temp = [] # Clear the array for the next iteration in the loop
            
            """ Contact Shear2 (Retina-Vitreous) """
            # Retina-Vitreous contact stress
            for Cs2_RV_i in fieldOutput[fieldObject_CSHEAR2_RV].values:
                temp.append(Cs2_RV_i.data*mult)
                
            S_Cs2_RV = np.sum(temp) # Sums up the list of stress from the "SET"
            Cshear2_RV_List.append(temp) # append the list of nodal values
            Cshear2_RV.append(S_Cs2_RV) # Adds the total stress to the stress-array by summing across each step
            temp = [] # Clear the array for the next iteration in the loop
            
            """ Contact Shear2 (Vitreous-Retina) """
            # Retina-Vitreous contact stress
            for Cs2_VR_i in fieldOutput[fieldObject_CSHEAR2_VR].values:
                temp.append(Cs2_VR_i.data*mult)
                
            S_Cs2_VR = np.sum(temp) # Sums up the list of stress from the "SET"
            Cshear2_VR_List.append(temp) # append the list of nodal values
            Cshear2_VR.append(S_Cs2_VR) # Adds the total stress to the stress-array by summing across each step
            temp = [] # Clear the array for the next iteration in the loop
            
            """ Contact ShearF (Retina-Vitreous) """
            # Retina-Vitreous contact stress
            for CsF_RV_i in fieldOutput[fieldObject_CSHEARF_RV].values:
                temp.append(CsF_RV_i.data*mult)
                
            S_CsF_RV = np.sum(temp) # Sums up the list of stress from the "SET"
            CshearF_RV_List.append(temp) # append the list of nodal values
            CshearF_RV.append(S_CsF_RV) # Adds the total stress to the stress-array by summing across each step
            temp = [] # Clear the array for the next iteration in the loop
            
            """ Contact ShearF (Vitreous-Retina) """
            # Retina-Vitreous contact stress
            for CsF_VR_i in fieldOutput[fieldObject_CSHEARF_VR].values:
                temp.append(CsF_VR_i.data*mult)
                
            S_CsF_VR = np.sum(temp) # Sums up the list of stress from the "SET"
            CshearF_VR_List.append(temp) # append the list of nodal values
            CshearF_VR.append(S_CsF_VR) # Adds the total stress to the stress-array by summing across each step
            temp = [] # Clear the array for the next iteration in the loop
            
        else:
            # append nans if not available
            CnormF_RV.append(np.nan)
            CnormF_VR.append(np.nan)
            Cpress_RV.append(np.nan)
            Cpress_RV_AVG.append(np.nan)
            Cpress_VR.append(np.nan)
            Cpress_VR_AVG.append(np.nan)
            Cshear1_RV.append(np.nan)
            Cshear1_VR.append(np.nan)
            Cshear2_RV.append(np.nan)
            Cshear2_VR.append(np.nan)
            CshearF_RV.append(np.nan)
            CshearF_VR.append(np.nan)
            
            CnormF_RV_List.append([np.nan, np.nan, np.nan])
            CnormF_VR_List.append([np.nan, np.nan, np.nan])
            Cpress_RV_List.append([np.nan, np.nan, np.nan])
            Cpress_VR_List.append([np.nan, np.nan, np.nan])
            Cshear1_RV_List.append([np.nan, np.nan, np.nan])
            Cshear1_VR_List.append([np.nan, np.nan, np.nan])
            Cshear2_RV_List.append([np.nan, np.nan, np.nan])
            Cshear2_VR_List.append([np.nan, np.nan, np.nan])
            CshearF_RV_List.append([np.nan, np.nan, np.nan])
            CshearF_VR_List.append([np.nan, np.nan, np.nan])

            print('No CPRESS... ** Updating with NANs')
        
        """ Contact Node Lists """
        R_V_SetNodeNames = []
        V_R_SetNodeNames = []
        for i, NodeLabeli in enumerate(odbInstance[part_R].nodeSets['R_V_SET'].nodes):
            R_V_SetNodeNames.append(NodeLabeli.label)
        
        for i, NodeLabeli in enumerate(odbInstance[part_V].nodeSets['V_R_SET'].nodes):
            V_R_SetNodeNames.append(NodeLabeli.label)
        
        """ Reaction forces """
        # fieldObject_RF = fieldOutput['RF'] # reaction forces
        # # Glue-Retina RP set-forces
        # Reaction_Forces = fieldObject_RF.getSubset(region=odbInstance[part_G].nodeSets['G_RP_SET'])
        # for RFi in Reaction_Forces.values: # Loops over each node in the "SET" defined by the reaction force
        #     RFi_vec = [RFi.data[0], RFi.data[1], RFi.data[2]]
        #     # Find the component in the direction of the load cell
        #     temp.append(np.dot(RFi_vec, LoadCellDirection)) # Creates a list of reaction forces in the "SET"
            
        # SRF = np.sum(temp) # Sums up the list of reaction forces from the "SET"
        # RF.append(SRF) # Adds the total reaction force to the RF-array by summing across each step
        # temp = [] # Clear the array for the next iteration in the loop
        
        fieldObject_RF = fieldOutput['RF'] # reaction forces
        if simplified == False:
            # If Simp is not in the title
            
            # Glue-Retina G_RP_Set Reaction forces
            Reaction_Forces = fieldObject_RF.getSubset(region=odbInstance[part_G].nodeSets['G_RP_SET'])
        
        elif simplified == True:
            
            # Retina R_G_Set Reaction forces
            Reaction_Forces = fieldObject_RF.getSubset(region=odbInstance[part_R].nodeSets['R_G_SET'])
        
        else:
            print('Error in RF output')
        
        for RFi in Reaction_Forces.values: # Loops over each node in the "SET" defined by the reaction force
            RFxi = RFi.data[0]
            RFyi = RFi.data[1]
            RFzi = RFi.data[2]
            RFi_vec = [RFxi, RFyi, RFzi]
            
            # Find the component in the direction of the load cell
            temp.append(np.dot(RFi_vec, LoadCellDirection)*mult) # Creates a list of reaction forces in the "SET"
            tempx.append(RFxi) # X reaction forces along the R_G_SET
            tempy.append(RFyi) # Y reaction forces along the R_G_SET
            tempz.append(RFzi) # Z reaction forces along the R_G_SET
                
        SRF = np.sum(temp) # Sums up the list of reaction forces from the "SET"
        RF.append(SRF) # Adds the total reaction force to the RF-array by summing across each step
        temp = [] # Clear the array for the next iteration in the loop
        
        SRFX = np.sum(tempx)
        RFx.append(SRFX)
        
        SRFY = np.sum(tempy)
        RFy.append(SRFY)
        
        SRFZ = np.sum(tempz)
        RFz.append(SRFZ)
        
        """ Nodal Forces """
        # Forces at the nodes of an element from both the hourglass and the regular 
        # deformation modes of that element (negative of the internal forces in 
        # the global coordinate system). The specified position in data and results 
        # file requests is ignored.
        
        if fieldOutput.has_key('NFORC1') == 1: # Searches if the repository has the value
            fieldObject_NFORC1 = fieldOutput['NFORC1'] # Normal force 1
            fieldObject_NFORC2 = fieldOutput['NFORC2'] # Normal force 2
            fieldObject_NFORC3 = fieldOutput['NFORC3'] # Normal force 3
            
            # Retina nodal forces on the glue interface
            nodeSet_R_G_SET = odbInstance[part_R].nodeSets['R_G_SET']
            NF1 = fieldObject_NFORC1.getSubset(region=nodeSet_R_G_SET)
            NF2 = fieldObject_NFORC2.getSubset(region=nodeSet_R_G_SET)
            NF3 = fieldObject_NFORC3.getSubset(region=nodeSet_R_G_SET)
            
            for NFi in range(len(NF1.values)): # Loops over each node in the "SET" defined by the reaction force
                NFi_vec = [NF1.values[NFi].data, NF2.values[NFi].data, NF3.values[NFi].data]
                NFi_veclabel = [NF1.values[NFi].nodeLabel, NF1.values[NFi].data, NF2.values[NFi].nodeLabel, NF2.values[NFi].data, NF3.values[NFi].nodeLabel, NF3.values[NFi].data]
                # Find the component in the direction of the load cell
                temp.append(np.dot(NFi_vec, LoadCellDirection)*mult) # Creates a list of reaction forces in the "SET"
            
            SNf = np.sum(temp) # Sums up the list of reaction forces from the "SET"
            Nforc.append(SNf*-1) # Adds the total reaction force to the RF-array by summing across each step (negative indicates the direction, which is opposite of tension when -1)
            temp = [] # Clear the array for the next iteration in the loop
        else:
            Nforc.append(0)
            print('No NFORC... ** Updating with 0')
        
    """ Loop over the history outputs"""
    # odb.steps[disp_step].historyRegions.keys() List all of the items in the dictionary
    odbHistoryRegion = odb.steps[disp_step].historyRegions
    odbHistAssem = 'Assembly ASSEMBLY'
    Assembly = odbHistoryRegion[odbHistAssem]
    
    # Energy output
    ALLIE_KE = Assembly.historyOutputs.keys()[0]
    Hist_ELEM = Assembly.historyOutputs.keys()[1]
    Whole_Model_Energy = Assembly.historyOutputs
    Internal_Energy = Whole_Model_Energy.keys()[0] # Internal energy
    Kinetic_Energy = Whole_Model_Energy.keys()[1] # Kintic energy
    for i, j in enumerate(Whole_Model_Energy[Internal_Energy].data):
        Hist_Time.append(j[0]) # History Output Time Array
        IE.append(j[1]) # Internal Energy
        KE.append(Whole_Model_Energy[Kinetic_Energy].data[i][1]) # Kinetic Energy
    
    if BondStatus == True:
    # if jobName.find('VRTie') == -1:
        # Figure out how to extract these for each node in the connected set
        """ Bond Loads """
        # This is an array of bond load per node
        V_R_SetNodeLength = len(odbInstance[part_V].nodeSets['V_R_SET'].nodes) # length of the V_R_Set node list
        BondNodeNames = odbHistoryRegion.keys()[-V_R_SetNodeLength:]
        
        # array of bond status and bond load
        BondStat = []
        BondLoad = []
        
        # used for iterating and clearing
        temp1 = []
        temp2 = []
    
        # loop over the length of the BondStat/BondLoad list
        for m,n in enumerate(odbHistoryRegion[BondNodeNames[0]].historyOutputs['BONDSTAT'].data):
            #  loop over the length of the bond node list and append each time step
            for i,BondNodeNames_i in enumerate(BondNodeNames):
                temp1.append(odbHistoryRegion[BondNodeNames_i].historyOutputs['BONDSTAT'].data[m][1]    )
                temp2.append(odbHistoryRegion[BondNodeNames_i].historyOutputs['BONDLOAD'].data[m][1]*mult)
            
            # build the arrays for BondStat/BondLoad
            BondStat.append(temp1)
            BondLoad.append(temp2)
            
            # clear the arrays
            temp1 = []
            temp2 = []
        else:
            print('No bonding,VR interface is tied')
    
    # # Contact
    # odbHistElementSetPIBATCH = odbHistoryRegion.keys()[1] # ElementSet PIBATCH
    # elementSetPIBATCH = odbHistoryRegion[odbHistElementSetPIBATCH]
    # eC = elementSetPIBATCH.historyOutputs # Element contact
    
    # if jobName.find('Si') == -1:
    #     # If Simp is not in the title
    #     cAreaCP_RG = eC.keys()[0]
    #     cAreaCP_RV = eC.keys()[1]
    #     cAreaCP_GR = eC.keys()[2]
    #     cAreaCP_VR = eC.keys()[3]
    #     CFN1CP_RG = eC.keys()[4]
    #     CFN1CP_RV = eC.keys()[5]
    #     CFN1CP_GR = eC.keys()[6]
    #     CFN1CP_VR = eC.keys()[7]
    #     CFN2CP_RG = eC.keys()[8]
    #     CFN2CP_RV = eC.keys()[9]
    #     CFN2CP_GR = eC.keys()[10]
    #     CFN2CP_VR = eC.keys()[11]
    #     CFN3CP_RG = eC.keys()[12]
    #     CFN3CP_RV = eC.keys()[13]
    #     CFN3CP_GR = eC.keys()[14]
    #     CFN3CP_VR = eC.keys()[15]
    
    # elif jobName.find('Si') >= 0:
    #     # If simplification omit the tab and glue
    #     cAreaCP_RV = eC.keys()[0]
    #     cAreaCP_VR = eC.keys()[1]
    #     CFN1CP_RV = eC.keys()[2]
    #     CFN1CP_VR = eC.keys()[3]
    #     CFN2CP_RV = eC.keys()[4]
    #     CFN2CP_VR = eC.keys()[5]
    #     CFN3CP_RV = eC.keys()[6]
    #     CFN3CP_VR = eC.keys()[7]
    # else:
    #     print('Error in Hist Output Names')
    #Bond_Nodes = energyHistRegion.historyOutputs.keys()[2:-1]
    
    # Glue Reference point
    if simplified == False:
        # If Simp is not in the title
        
        odbHist_gRP = odbHistoryRegion.keys()[1]
        gRP_Hist = odbHistoryRegion[odbHist_gRP]
        gRP_Hist = gRP_Hist.historyOutputs
        gRP_HistRF1 = gRP_Hist.keys()[0]
        gRP_HistRF2 = gRP_Hist.keys()[1]
        gRP_HistRF3 = gRP_Hist.keys()[2]
        gRP_HistU1 = gRP_Hist.keys()[6]
        gRP_HistU2 = gRP_Hist.keys()[7]
        gRP_HistU3 = gRP_Hist.keys()[8]
        
    elif simplified == True:
        # If simplification, omit the tab and glue
        print('Simplification')
    else:
        print('Error in simplification')
    
    # for i,j in enumerate(Internal_Energy.data):
    #     Hist_Time.append(j[0]) # History Output Time Array
        
    #     # Energy array
    #     IE.append(j[1]*mult) # Internal Energy
    #     KE.append(Kinetic_Energy.data[i][1]*mult) # Kinetic Energy
        
    #     if jobName.find('Si') == -1:
    #         # If Simp is not in the title
            
    #         # Contact area arrays, not sure if these need to be multiplied by 2 (Check .ODB)
    #         CAreaCP_RG.append(eC[cAreaCP_RG].data[i][1])
    #         CAreaCP_GR.append(eC[cAreaCP_GR].data[i][1])
    #         CAreaCP_RV.append(eC[cAreaCP_RV].data[i][1])
    #         CAreaCP_VR.append(eC[cAreaCP_VR].data[i][1])
            
    #         # Create a vector for CP RG
    #         CFNCP_RG_vec = [eC[CFN1CP_RG].data[i][1], eC[CFN2CP_RG].data[i][1], eC[CFN3CP_RG].data[i][1]]
    #         # Find the component in the direction of the load cell
    #         CFNCP_RG.append(np.dot(CFNCP_RG_vec, LoadCellDirection)*mult)
            
    #         # Create a vector for CP GR
    #         CFNCP_GR_vec = [eC[CFN1CP_GR].data[i][1], eC[CFN2CP_GR].data[i][1], eC[CFN3CP_GR].data[i][1]]
    #         # Find the component in the direction of the load cell
    #         CFNCP_GR.append(np.dot(CFNCP_GR_vec, LoadCellDirection)*mult)
            
    #         # Create a vector for CP RV
    #         CFNCP_RV_vec = [eC[CFN1CP_RV].data[i][1], eC[CFN2CP_RV].data[i][1], eC[CFN3CP_RV].data[i][1]]
    #         # Find the component in the direction of the load cell
    #         CFNCP_RV.append(np.dot(CFNCP_RV_vec, LoadCellDirection)*mult)
            
    #         # Create a vector for CP VR
    #         CFNCP_VR_vec = [eC[CFN1CP_VR].data[i][1], eC[CFN2CP_VR].data[i][1], eC[CFN3CP_VR].data[i][1]]
    #         # Find the component in the direction of the load cell
    #         CFNCP_VR.append(np.dot(CFNCP_VR_vec, LoadCellDirection)*mult)
            
    #         # Create a vector for the Glue Reference point
    #         Glue_RP_RF_vec = [gRP_Hist[gRP_HistRF1].data[i][1], gRP_Hist[gRP_HistRF2].data[i][1], gRP_Hist[gRP_HistRF3].data[i][1]]
    #         # Find the component in the direction of the load cell
    #         Glue_RP_RF.append(np.dot(Glue_RP_RF_vec, LoadCellDirection)*mult)
        
    #     elif jobName.find('Si') >= 0:
    #         # Contact area arrays
    #         CAreaCP_RG.append(np.nan)
    #         CAreaCP_GR.append(np.nan)
    #         CAreaCP_RV.append(eC[cAreaCP_RV].data[i][1])
    #         CAreaCP_VR.append(eC[cAreaCP_VR].data[i][1])
            
    #         # Create a vector for CP RG
    #         CFNCP_RG_vec = [np.nan, np.nan, np.nan]
    #         # Find the component in the direction of the load cell
    #         CFNCP_RG.append(np.dot(CFNCP_RG_vec, LoadCellDirection)*mult)
            
    #         # Create a vector for CP GR
    #         CFNCP_GR_vec = [np.nan, np.nan, np.nan]
    #         # Find the component in the direction of the load cell
    #         CFNCP_GR.append(np.dot(CFNCP_GR_vec, LoadCellDirection)*mult)
            
    #         # Create a vector for CP RV
    #         CFNCP_RV_vec = [eC[CFN1CP_RV].data[i][1], eC[CFN2CP_RV].data[i][1], eC[CFN3CP_RV].data[i][1]]
    #         # Find the component in the direction of the load cell
    #         CFNCP_RV.append(np.dot(CFNCP_RV_vec, LoadCellDirection)*mult)
            
    #         # Create a vector for CP VR
    #         CFNCP_VR_vec = [eC[CFN1CP_VR].data[i][1], eC[CFN2CP_VR].data[i][1], eC[CFN3CP_VR].data[i][1]]
    #         # Find the component in the direction of the load cell
    #         CFNCP_VR.append(np.dot(CFNCP_VR_vec, LoadCellDirection)*mult)
            
    #         # Create a vector for the Glue Reference point
    #         Glue_RP_RF_vec = [np.nan, np.nan, np.nan]
    #         # Find the component in the direction of the load cell
    #         Glue_RP_RF.append(np.dot(Glue_RP_RF_vec, LoadCellDirection)*mult)
    #     else:
    #         print('Error in hist output data with simplification')
        
    # if jobName.find('VRTie') == -1: # if VRTie is not in the title
    #     """ Bond Loads """
    #     # This is an array of bond load per node
    #     V_R_SetNodeLength = len(odbInstance[part_V].nodeSets['V_R_SET'].nodes) # length of the V_R_Set node list
    #     BondNodeNames = odbHistoryRegion.keys()[-V_R_SetNodeLength:]
        
    #     # array of bond status and bond load
    #     BondStat = []
    #     BondLoad = []
        
    #     # used for iterating and clearing
    #     temp1 = []
    #     temp2 = []
    
    #     # loop over the length of the BondStat/BondLoad list
    #     for m,n in enumerate(odbHistoryRegion[BondNodeNames[0]].historyOutputs['BONDSTAT'].data):
    #         #  loop over the length of the bond node list and append each time step
    #         for i,BondNodeNames_i in enumerate(BondNodeNames):
    #             temp1.append(odbHistoryRegion[BondNodeNames_i].historyOutputs['BONDSTAT'].data[m][1])
    #             temp2.append(odbHistoryRegion[BondNodeNames_i].historyOutputs['BONDLOAD'].data[m][1])
            
    #         # build the arrays for BondStat/BondLoad
    #         BondStat.append(temp1)
    #         BondLoad.append(temp2)
            
    #         # clear the arrays
    #         temp1 = []
    #         temp2 = []
    #     else:
    #         print('No bond because the VR interface is tied')
            
    """ Specify folder name where the files go..."""
    folderName = jobName
    folder_sub_directory = 'Output'
    
    """ Print the odbFieldOutput Data """
    print("\nWriting out the load data...")
    filename = os.path.join(folderName, folder_sub_directory, 'output_Field_' 
                            + jobName + '.txt')
    outfile = open(filename,'w')
    
    Header = [] # Header information for the dataframe
    Header.append('frame')
    Header.append('Time [s]')
    Header.append('Reaction force dotted in y direction [N]')
    Header.append('Reaction force X [N]')
    Header.append('Reaction force Y [N]')
    Header.append('Reaction force Z [N]')
    Header.append('Sum NForc (Glue-Retina Set) [N]')
    Header.append('CnormF_RV [N]')
    Header.append('CnormF_VR [N]')
    Header.append('Cpress_RV [Pa]')
    Header.append('Cpress_VR [Pa]')
    Header.append('AVG_Cpress_RV_AVG [Pa]')
    Header.append('AVG_Cpress_VR_AVG [Pa]')
    Header.append('Cshear1_RV [Pa]')
    Header.append('Cshear1_VR [Pa]')
    Header.append('Cshear2_RV [Pa]')
    Header.append('Cshear2_VR [Pa]')
    Header.append('CshearF_RV [Pa]')
    Header.append('CshearF_VR [Pa]')
    Header.append('Glue Displacements [m]')
    Header.append('Bond Displacements [m]')
    lineWrite = '\t'.join(str(item) for item in Header)
    outfile.write(lineWrite)
    
    for i in frames:
        
        lineNums = []
        lineNums.append(time[i])
        lineNums.append(RF[i])
        lineNums.append(RFx[i])
        lineNums.append(RFy[i])
        lineNums.append(RFz[i])
        lineNums.append(Nforc[i])
        lineNums.append(CnormF_RV[i])
        lineNums.append(CnormF_VR[i])
        lineNums.append(Cpress_RV[i])
        lineNums.append(Cpress_VR[i])
        lineNums.append(Cpress_RV_AVG[i])
        lineNums.append(Cpress_VR_AVG[i])
        lineNums.append(Cshear1_RV[i])
        lineNums.append(Cshear1_VR[i])
        lineNums.append(Cshear2_RV[i])
        lineNums.append(Cshear2_VR[i])
        lineNums.append(CshearF_RV[i])
        lineNums.append(CshearF_VR[i])
        lineNums.append(U_top[i])
        lineNums.append(Bond_disp[i])
        
        # format the list to have a float with twenty decimal places
        # Add floats
        formatted_list = ['{:.20f}'.format(item) for item in lineNums]
        line = '\n' + '{}\t'.format(i) + '\t'.join(str(item) for item in 
                                                   formatted_list)
        outfile.write(line)
        
    outfile.close()
    
    print("\nDone!")
    print("\nThe output file will be named '{}".format(filename) + "'")
    print("\nIt will be in the same working directory as your Abaqus model\n")
    
    """ Print the odbHistoryOutput Data """
    print("\nWriting out the History Output data...")
    filename = os.path.join(folderName, 'Output', 'output_History_' + 
                            jobName + '.txt')
    outfile = open(filename, 'w')
    
    Header = []
    Header.append('frame')
    Header.append('Time [s]')
    Header.append('Internal Energy [J]')
    Header.append('Kinetic Energy [J]')
    lineWrite = '\t'.join(str(item) for item in Header)
    outfile.write(lineWrite)
    
    for i, j in enumerate(Hist_Time):
        line = []
        line.append('{}'.format(i)) # Integer for frame number
        line.append('{:.10f}'.format(j))
        line.append('{:.30f}'.format(IE[i]))
        line.append('{:.30f}'.format(KE[i]))
        lineWrite = '\n' + '\t'.join(str(item) for item in line)
        outfile.write(lineWrite)
        
    outfile.close()
    
    print("\nDone!")
    print("\nThe output file will be named '{}".format(filename) + "'")
    print("\nIt will be in the same working directory as your Abaqus model\n")
    
    if BondStatus == True:
    # if jobName.find('VRTie') == -1: # if VRTie is not in the jobName
        """ Cube Info Plots """
        """ Print the odbHistoryOutput BondStat Data """
        print("\nWriting out the History Output Bond data...")
        filename = os.path.join(folderName, 'Output', 'BONDSTAT_' + jobName + '.txt')
        outfile = open(filename,'w')
        outfile.write('Time (s)\t' + '\t'.join(str(item) for item in BondNodeNames))
        for i,j in enumerate(BondStat):
            outfile.write('\n')
            tempList = [Hist_Time[i]]
            for k in list(j):
                tempList.append(k)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus model\n")
        
        """ Print the odbHistoryOutput BondLoad Data """
        print("\nWriting out the History Output Bond data...")
        filename = os.path.join(folderName, 'Output', 'BONDLOAD_' + jobName + '.txt')
        outfile = open(filename,'w')
        outfile.write('Time (s)\t' + '\t'.join(str(item) for item in BondNodeNames))
        for i,j in enumerate(BondLoad):
            outfile.write('\n')
            tempList = [Hist_Time[i]]
            for k in list(j):
                tempList.append(k)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus model\n")
        
        """ Print the odbFieldOutput CnormF_RV Data """
        print("\nWriting out the Field Output CnormF_RV data...")
        filename = os.path.join(folderName, 'Output', 'CnormF_RV_' + jobName + '.txt')
        outfile = open(filename,'w')
        xyz = ['X','Y','Z']
        header = []
        for R_V_SetXYZi in list(R_V_SetNodeNames):
            for m in range(3):
                header.append('R-' + str(R_V_SetXYZi) + xyz[m])
        outfile.write('Time (s)\t' + '\t'.join(item for item in header))
        for i,Nodei in enumerate(CnormF_RV_List):
            outfile.write('\n')
            tempList = [time[i]]
            for XYZ in list(Nodei):
                for XYZi in list(XYZ):
                    tempList.append(XYZi)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus model\n")
    
        """ Print the odbFieldOutput CnormF_VR Data """
        print("\nWriting out the Field Output CnormF_VR data...")
        filename = os.path.join(folderName, 'Output', 'CnormF_VR_' + jobName + '.txt')
        outfile = open(filename,'w')
        xyz = ['X','Y','Z']
        header = []
        for V_R_SetXYZi in list(V_R_SetNodeNames):
            for m in range(3):
                header.append('V-' + str(V_R_SetXYZi) + xyz[m])
        outfile.write('Time (s)\t' + '\t'.join(item for item in header))
        for i,Nodei in enumerate(CnormF_VR_List):
            outfile.write('\n')
            tempList = [time[i]]
            for XYZ in list(Nodei):
                for XYZi in list(XYZ):
                    tempList.append(XYZi)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus model\n")
        
        """ Print the odbFieldOutput Cpress_RV Data """
        print("\nWriting out the Field Output Cpress_RV data...")
        filename = os.path.join(folderName, 'Output', 'Cpress_RV_' + jobName + '.txt')
        outfile = open(filename,'w')
        header = []
        for R_V_SetXYZi in list(R_V_SetNodeNames):
            header.append('R-' + str(R_V_SetXYZi))
        outfile.write('Time (s)\t' + '\t'.join(item for item in header))
        for i,Nodei in enumerate(Cpress_RV_List):
            outfile.write('\n')
            tempList = [time[i]]
            for ni in list(Nodei):
                tempList.append(ni)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus model\n")
        
        """ Print the odbFieldOutput Cpress_VR Data """
        print("\nWriting out the Field Output Cpress_VR data...")
        filename = os.path.join(folderName, 'Output', 'Cpress_VR_' + jobName + '.txt')
        outfile = open(filename,'w')
        header = []
        for V_R_SetXYZi in list(V_R_SetNodeNames):
            header.append('V-' + str(V_R_SetXYZi))
        outfile.write('Time (s)\t' + '\t'.join(item for item in header))
        for i,Nodei in enumerate(Cshear1_RV_List):
            outfile.write('\n')
            tempList = [time[i]]
            for ni in list(Nodei):
                tempList.append(ni)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus model\n")
        
        """ Print the odbFieldOutput Cshear1_RV Data """
        print("\nWriting out the Field Output Cshear1_RV data...")
        filename = os.path.join(folderName, 'Output', 'Cshear1_RV_' + jobName + '.txt')
        outfile = open(filename,'w')
        header = []
        for R_V_SetXYZi in list(R_V_SetNodeNames):
            header.append('R-' + str(R_V_SetXYZi))
        outfile.write('Time (s)\t' + '\t'.join(item for item in header))
        for i,Nodei in enumerate(Cshear1_RV_List):
            outfile.write('\n')
            tempList = [time[i]]
            for ni in list(Nodei):
                tempList.append(ni)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus model\n")
        
        """ Print the odbFieldOutput Cshear1_VR Data """
        print("\nWriting out the Field Output Cshear1_VR data...")
        filename = os.path.join(folderName, 'Output', 'Cshear1_VR_' + jobName + '.txt')
        outfile = open(filename,'w')
        header = []
        for V_R_SetXYZi in list(V_R_SetNodeNames):
            header.append('V-' + str(V_R_SetXYZi))
        outfile.write('Time (s)\t' + '\t'.join(item for item in header))
        for i,Nodei in enumerate(Cshear1_VR_List):
            outfile.write('\n')
            tempList = [time[i]]
            for ni in list(Nodei):
                tempList.append(ni)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus model\n")
        
        """ Print the odbFieldOutput Cshear2_RV Data """
        print("\nWriting out the Field Output Cshear2_RV data...")
        filename = os.path.join(folderName, 'Output', 'Cshear2_RV_' + jobName + '.txt')
        outfile = open(filename,'w')
        header = []
        for R_V_SetXYZi in list(R_V_SetNodeNames):
            header.append('R-' + str(R_V_SetXYZi))
        outfile.write('Time (s)\t' + '\t'.join(item for item in header))
        for i,Nodei in enumerate(Cshear2_RV_List):
            outfile.write('\n')
            tempList = [time[i]]
            for ni in list(Nodei):
                tempList.append(ni)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus model\n")
        
        """ Print the odbFieldOutput Cshear2_VR Data """
        print("\nWriting out the Field Output Cshear2_VR data...")
        filename = os.path.join(folderName, 'Output', 'Cshear2_VR_' + jobName + '.txt')
        outfile = open(filename,'w')
        header = []
        for V_R_SetXYZi in list(V_R_SetNodeNames):
            header.append('V-' + str(V_R_SetXYZi))
        outfile.write('Time (s)\t' + '\t'.join(item for item in header))
        for i,Nodei in enumerate(Cshear2_VR_List):
            outfile.write('\n')
            tempList = [time[i]]
            for ni in list(Nodei):
                tempList.append(ni)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus model\n")
    
        """ Print the odbFieldOutput CshearF_RV Data """
        print("\nWriting out the Field Output CshearF_RV data...")
        filename = os.path.join(folderName, 'Output', 'CshearF_RV_' + jobName + '.txt')
        outfile = open(filename,'w')
        xyz = ['X','Y','Z']
        header = []
        for R_V_SetXYZi in list(R_V_SetNodeNames):
            for m in range(3):
                header.append('R-' + str(R_V_SetXYZi) + xyz[m])
        outfile.write('Time (s)\t' + '\t'.join(item for item in header))
        for i,Nodei in enumerate(CshearF_RV_List):
            outfile.write('\n')
            tempList = [time[i]]
            for XYZ in list(Nodei):
                for XYZi in list(XYZ):
                    tempList.append(XYZi)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus model\n")
        
        """ Print the odbFieldOutput CshearF_VR Data """
        print("\nWriting out the Field Output CshearF_VR data...")
        filename = os.path.join(folderName, 'Output', 'CshearF_VR_' + jobName + '.txt')
        outfile = open(filename,'w')
        xyz = ['X','Y','Z']
        header = []
        for V_R_SetXYZi in list(V_R_SetNodeNames):
            for m in range(3):
                header.append('V-' + str(V_R_SetXYZi) + xyz[m])
        outfile.write('Time (s)\t' + '\t'.join(item for item in header))
        for i,Nodei in enumerate(CshearF_VR_List):
            outfile.write('\n')
            tempList = [time[i]]
            for XYZ in list(Nodei):
                for XYZi in list(XYZ):
                    tempList.append(XYZi)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus model\n")
    
    else:
        print('No bonding because the VR Interface is tied')
    return

# Run the function
data_extract(jobName)