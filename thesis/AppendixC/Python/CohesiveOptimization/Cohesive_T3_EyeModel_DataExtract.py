# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:36:37 2021

@author: Kiffer Creveling
Instructions:
    1) Save this script in a folder containing your ODB file
    2) Open a command window and navigate to your directory containing this 
    script and your ODB file
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
# Arguments from the previous script
script =             sys.argv[0]
jobName =            sys.argv[1]
gravity =       eval(sys.argv[2]) # True/False
symmetry =      eval(sys.argv[3]) # True/False
simplified =    eval(sys.argv[4]) # True/False
DMGInitiation = eval(sys.argv[5]) # True/False
DMGEvolution =  eval(sys.argv[6]) # True/False # not used in the extraction

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
    
    Stress = [] # Stress
    
    CSDMG = [] # Damage variable for cohesive surfaces in general contact.
    # Maximum stress-based damage initiation criterion for cohesive surfaces 
    # in general contact.
    CSMAXSCRT = [] 
    
    CSDMG_List = [] # values
    CSMAXSCRT_List = [] # values
    
    CSDMG_Nodes = [] # nodes
    CSMAXSCRT_Nodes = [] # nodes
    
    frames = [] # List of frames
    time = [] # Time array
    
    # Used for reaction force simplicity further in the code
    # Temporary array used for iterating (Clears after each iteration)
    temp = []
    tempx = []
    tempy = []
    tempz = []
    
    """ History Output data arrays """
    Hist_Time = []
    IE = []
    KE = []
    
    """ Loop over the field outputs"""
    # determines the step in the abaqus odb file (typically displacement)
    step = odb.steps.keys()
    
    if gravity != True:
        disp_step = step[0] # Defines the step as a variable name
    else:
        # Step that includes the gravity kinetic energy settling
        disp_step = step[1]
    
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
            # Parts
            E = odbInstance.keys(0)[0]
            G = odbInstance.keys(0)[1]
            R = odbInstance.keys(0)[2]
            T = odbInstance.keys(0)[3]
            V = odbInstance.keys(0)[4]
            
        elif simplified == True:
            # If simplification exists, omit the glue & tab
            E = odbInstance.keys(0)[0]
            R = odbInstance.keys(0)[1]
            V = odbInstance.keys(0)[2]
        else:
            print('Error in part definitions')
        
        """ Nodal displacements """
        fO_U = fieldOutput['U'] # displacements
        
        if simplified == False:
            # If Simp is not in the title
            
            # Glue
            Displacements = fO_U.getSubset(region=odbInstance[G]
                                                    .nodeSets['G_RP_SET'])
            # Loops over each node in the "SET" defined by the displacement
            for Uyi in Displacements.values:
                Uyi_vec = [Uyi.data[0], Uyi.data[1], Uyi.data[2]]
                # Find the magnitude
                # Creates a list of displacements in the "SET"
                temp.append(np.dot(Uyi_vec, LoadCellDirection))
                
            # Sums up the list of displacements from the "SET"
            SU = np.sum(temp)
            # Divide by the number of nodes in the set to get average
            AvgU_top = SU/len(temp)
            # Adds the total displacement to the U-array by summing across 
            # each step
            U_top.append(AvgU_top)
            temp = [] # Clear the array for the next iteration in the loop
            
        elif simplified == True:
            # If simplification exists, omit the values
            
            Displacements = fO_U.getSubset(region=odbInstance[R]
                                                    .nodeSets['R_G_SET'])
            # Loops over each node in the "SET" defined by the displacement
            for Uyi in Displacements.values:
                Uyi_vec = [Uyi.data[0], Uyi.data[1], Uyi.data[2]]
                # Find the magnitude
                # Creates a list of displacements in the "SET"
                temp.append(np.dot(Uyi_vec, LoadCellDirection))
                
            # Sums up the list of displacements from the "SET"
            SU = np.sum(temp)
            # Divide by the number of nodes in the set to get average
            AvgU_top = SU/len(temp)
            # Adds the total displacement to the U-array by summing across 
            # each step
            U_top.append(AvgU_top)
            temp = [] # Clear the array for the next iteration in the loop
            
        else:
            print('Error in nodal displacements')
        
        """ Bond Distance """
        Displacements = fO_U.getSubset(region=odbInstance[R]
                                                .nodeSets['R_V_SET'])
        # Loops over each node in the "SET" defined by the displacement
        for Uyi in Displacements.values:
            Uyi_vec = [Uyi.data[0], Uyi.data[1], Uyi.data[2]]
            # Find the magnitude
            # Creates a list of displacements in the "SET"
            temp.append(np.dot(Uyi_vec, LoadCellDirection))
            
        # Sums up the list of displacements from the "SET"
        SU = np.sum(temp)
        # Divide by the number of nodes in the set to get average
        AvgR_bot = SU/len(temp)
        # Adds the total displacement to the U-array by summing across 
        # each step
        R_bot.append(AvgR_bot)
        temp = [] # Clear the array for the next iteration in the loop
            
        Displacements = fO_U.getSubset(region=odbInstance[V]
                                                .nodeSets['V_R_SET'])
        # Loops over each node in the "SET" defined by the displacement
        for Uyi in Displacements.values:
            Uyi_vec = [Uyi.data[0], Uyi.data[1], Uyi.data[2]]
            # Find the magnitude
            # Creates a list of displacements in the "SET"
            temp.append(np.dot(Uyi_vec, LoadCellDirection))
            
        # Sums up the list of displacements from the "SET"
        SU = np.sum(temp)
        # Divide by the number of nodes in the set to get average
        AvgV_top = SU/len(temp)
        # Adds the total displacement to the U-array by summing across 
        # each step
        V_top.append(AvgV_top)
        temp = [] # Clear the array for the next iteration in the loop
        
        # average difference in nodal positions between the *bonded surfaces
        Bond_disp.append(AvgR_bot - AvgV_top)
        
        """ Cohesive Info """
        if DMGInitiation == True:
            # fieldObject_CSMAXSCRT
            fO_CMS = fieldOutput['CSMAXSCRT General_Contact_Domain']
            
            # Specify only the bonded interface
            BONDED_Surface_R_CMS = fO_CMS.getSubset(region=odbInstance[R]
                                                    .nodeSets['R_V_SET'])
            BONDED_Surface_V_CMS = fO_CMS.getSubset(region=odbInstance[V]
                                                    .nodeSets['V_R_SET'])
            
            """ Contact initiation for cohesive surfaces """
            # Retina-Vitreous cohesive initiation value
            
            # Loop over all retina nodes
            for CSMAXSCRT_i in BONDED_Surface_R_CMS.values:
                temp.append(CSMAXSCRT_i.data) # nodal value
                if frame == 0:
                    CSMAXSCRT_Nodes.append(CSMAXSCRT_i.nodeLabel)
            
            # Loop over all vitreous nodes
            for CSMAXSCRT_i in BONDED_Surface_V_CMS.values:
                temp.append(CSMAXSCRT_i.data) # nodal value
                if frame == 0:
                    CSMAXSCRT_Nodes.append(CSMAXSCRT_i.nodeLabel)
            
            # Mean of the list of initiation values from the "SET"
            Mean_CMS = np.mean(temp)
            # append the list of nodal values
            CSMAXSCRT_List.append(temp)
            # Adds the average value to the array by summing across each step
            CSMAXSCRT.append(Mean_CMS)
            temp = [] # Clear the array for the next iteration in the loop
            
        else:
            print('No cohesive initiation info to update... ** Updating ' + 
                  'with nans')
            CSMAXSCRT_List.append(np.nan)
            CSMAXSCRT.append(np.nan)
            CSMAXSCRT_Nodes.append(np.nan)
        
        if DMGEvolution == True:
            # fieldObject_CSDMG
            fO_CDG = fieldOutput['CSDMG General_Contact_Domain']
            
            # Specify only the bonded interface
            BONDED_Surface_R_CSDMG = fO_CDG.getSubset(region=odbInstance[R]
                                                      .nodeSets['R_V_SET'])
            BONDED_Surface_V_CSDMG = fO_CDG.getSubset(region=odbInstance[V]
                                                      .nodeSets['V_R_SET'])
            
            """ Contact damage for cohesive surfaces """
            # Retina-Vitreous cohesive damage value
            
            # Loop over all retina nodes
            for CSDMG_i in BONDED_Surface_R_CSDMG.values:
                temp.append(CSDMG_i.data)
                if frame == 0:
                    CSDMG_Nodes.append(CSDMG_i.nodeLabel)
            
            # Loop over all vitreous nodes
            for CSDMG_i in BONDED_Surface_V_CSDMG.values:
                temp.append(CSDMG_i.data)
                if frame == 0:
                    CSDMG_Nodes.append(CSDMG_i.nodeLabel)
            
            # Mean of the list of initiation values from the "SET"
            Mean_CSDMG = np.mean(temp)
            # append the list of nodal values
            CSDMG_List.append(temp)
            # Adds the average value to the array by summing across each step
            CSDMG.append(Mean_CSDMG)
            temp = [] # Clear the array for the next iteration in the loop
        else:
            print('No cohesive damage info to update... ** Updating with nans')
            CSDMG_List.append(np.nan)
            CSDMG.append(np.nan)
            CSDMG_Nodes.append(np.nan)
        
        """ Contact Node Lists """
        R_V_SetNodeNames = []
        V_R_SetNodeNames = []
        for i, NodeLabeli in enumerate(odbInstance[R]
                                       .nodeSets['R_V_SET'].nodes):
            R_V_SetNodeNames.append(NodeLabeli.label)
        
        for i, NodeLabeli in enumerate(odbInstance[V]
                                       .nodeSets['V_R_SET'].nodes):
            V_R_SetNodeNames.append(NodeLabeli.label)
        
        """ Reaction forces """
        fO_RF = fieldOutput['RF'] # reaction forces
        if simplified == False:
            # If Simp is not in the title
            
            # Glue-Retina G_RP_Set Reaction forces
            Reaction_Forces = fO_RF.getSubset(region=odbInstance[G]
                                              .nodeSets['G_RP_SET'])
        
        elif simplified == True:
            
            # Retina R_G_Set Reaction forces
            Reaction_Forces = fO_RF.getSubset(region=odbInstance[R]
                                              .nodeSets['R_G_SET'])
        
        else:
            print('Error in RF output')
        
        # Loops over each node in the "SET" defined by the reaction force
        for RFi in Reaction_Forces.values:
            RFxi = RFi.data[0]
            RFyi = RFi.data[1]
            RFzi = RFi.data[2]
            RFi_vec = [RFxi, RFyi, RFzi]
            
            # Find the component in the direction of the load cell
            # Creates a list of reaction forces in the "SET"
            temp.append(np.dot(RFi_vec, LoadCellDirection)*mult)
            tempx.append(RFxi*mult) # X reaction forces along the R_G_SET
            tempy.append(RFyi*mult) # Y reaction forces along the R_G_SET
            tempz.append(RFzi*mult) # Z reaction forces along the R_G_SET
                
        SRF = np.sum(temp) # Sums up the list of reaction forces from the "SET"
        # Adds the total reaction force to the RF-array by summing across 
        # each step
        RF.append(SRF)
        temp = [] # Clear the array for the next iteration in the loop
        
        SRFX = np.sum(tempx)
        RFx.append(SRFX)
        
        SRFY = np.sum(tempy)
        RFy.append(SRFY)
        
        SRFZ = np.sum(tempz)
        RFz.append(SRFZ)
        
        """ Nodal Forces """
        ''' Forces at the nodes of an element from both the hourglass and the 
        regular deformation modes of that element (negative of the internal 
        forces in the global coordinate system). The specified position in 
        data and results file requests is ignored.'''
        
        # Searches if the repository has the value
        if fieldOutput.has_key('NFORC1') == 1:
            fO_NFORC1 = fieldOutput['NFORC1'] # Normal force 1
            fO_NFORC2 = fieldOutput['NFORC2'] # Normal force 2
            fO_NFORC3 = fieldOutput['NFORC3'] # Normal force 3
            
            # Retina nodal forces on the glue interface
            nodeSet_R_G_SET = odbInstance[R].nodeSets['R_G_SET']
            NF1 = fO_NFORC1.getSubset(region=nodeSet_R_G_SET)
            NF2 = fO_NFORC2.getSubset(region=nodeSet_R_G_SET)
            NF3 = fO_NFORC3.getSubset(region=nodeSet_R_G_SET)
            
            # Loops over each node in the "SET" defined by the reaction force
            for NFi in range(len(NF1.values)):
                NFi_vec = [NF1.values[NFi].data, 
                           NF2.values[NFi].data, 
                           NF3.values[NFi].data]
                NFi_veclabel = [NF1.values[NFi].nodeLabel, 
                                NF1.values[NFi].data, 
                                NF2.values[NFi].nodeLabel, 
                                NF2.values[NFi].data, 
                                NF3.values[NFi].nodeLabel, 
                                NF3.values[NFi].data]
                # Find the component in the direction of the load cell
                # Creates a list of reaction forces in the "SET"
                temp.append(np.dot(NFi_vec, LoadCellDirection)*mult)
            
            # Sums up the list of reaction forces from the "SET"
            SNf = np.sum(temp)
            # Adds the total reaction force to the RF-array by summing across 
            # each step (negative indicates the direction, which is opposite 
            # of tension when -1)
            Nforc.append(SNf*-1)
            temp = [] # Clear the array for the next iteration in the loop
        else:
            Nforc.append(0)
            print('No NFORC... ** Updating with 0')
        
        """ Stress """
        fO_S = fieldOutput['S'] # stress
        # Glue-Retina set-forces
        # Loops over each node in the "SET" defined by the reaction force
        for Si in fO_S.values:
            stress_vec = [Si.data[0], Si.data[1], Si.data[2]]
            # Append the component of stress in the load cell direction
            Stress.append(np.dot(stress_vec, LoadCellDirection))
    
    # In[History Output]
    """ Loop over the history outputs"""
    # List all of the items in the dictionary
    # odb.steps[disp_step].historyRegions.keys()
    odbHistoryRegion = odb.steps[disp_step].historyRegions
    odbHistAssem = 'Assembly ASSEMBLY'
    Assembly = odbHistoryRegion[odbHistAssem]
    
    # Energy output
    ALLIE_KE = Assembly.historyOutputs.keys()[0]
    Hist_ELEM = Assembly.historyOutputs.keys()[1]
    Whole_Model_Energy = Assembly.historyOutputs
    Internal_Energy = Whole_Model_Energy.keys()[0] # Internal energy
    Kinetic_Energy = Whole_Model_Energy.keys()[1] # Kintic energy
    for i,j in enumerate(Whole_Model_Energy[Internal_Energy].data):
        Hist_Time.append(j[0]) # History Output Time Array
        IE.append(j[1]) # Internal Energy
        # Kinetic Energy
        KE.append(Whole_Model_Energy[Kinetic_Energy].data[i][1])
    
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
    
    # In[Print Field Outputs]
    """ Specify folder name where the files go..."""
    folderName = jobName
    folder_sub_directory = 'Output'
    
    """ Print the odbFieldOutput Data """
    print("\nWriting out the load data...")
    filename = os.path.join(folderName, folder_sub_directory, 
                            'output_Field_' + jobName + '.txt')
    outfile = open(filename,'w')
    
    Header = [] # Header information for the dataframe
    Header.append('frame')
    Header.append('Time [s]')
    Header.append('Reaction force dotted in y direction [N]')
    Header.append('Reaction force X [N]')
    Header.append('Reaction force Y [N]')
    Header.append('Reaction force Z [N]')
    Header.append('Sum Nodal Force [N]')
    Header.append('Glue Displacements [m]')
    Header.append('Bond Displacements [m]')
    Header.append('Stress [Pa]')
    Header.append('AVG CSMAXSCRT')
    Header.append('AVG CSDMG')
    
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
        lineNums.append(U_top[i])
        lineNums.append(Bond_disp[i])
        lineNums.append(Stress[i])
        lineNums.append(CSMAXSCRT[i])
        lineNums.append(CSDMG[i])
        
        # format the list to have a float with twenty decimal places
        # Add floats
        formatted_list = ['{:.20f}'.format(item) for item in lineNums]
        line = '\n' + '{}\t'.format(i) + '\t'.join(str(item) 
                                                   for item in formatted_list)
        outfile.write(line)
        
    outfile.close()
    
    print("\nDone!")
    print("\nThe output file will be named '{}".format(filename) + "'")
    print("\nIt will be in the same working directory as your Abaqus" + 
          " model\n")
    
    # In[Print History Output]
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
    print("\nIt will be in the same working directory as your Abaqus" + 
          " model\n")
    # In[DMG Criteria]
    if DMGInitiation == True:
        """ Print the CSMAXSCRT Data """
        print("\nWriting out the Field Output CSMAXSCRT data...")
        filename = os.path.join(folderName, 'Output', 'CSMAXSCRT_' + 
                                jobName + '.txt')
        outfile = open(filename, 'w')
        outfile.write('Time (s)\t' + '\t'.join(str(item) 
                                               for item in CSMAXSCRT_Nodes))
        for i, j in enumerate(CSMAXSCRT_List):
            outfile.write('\n')
            tempList = [time[i]]
            for k in list(j):
                tempList.append(k)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus" + 
              " model\n")
    if DMGEvolution == True:
        """ Print the CSDMG Data """
        print("\nWriting out the Field Output CSDMG data...")
        filename = os.path.join(folderName, 'Output', 'CSDMG_' + 
                                jobName + '.txt')
        outfile = open(filename, 'w')
        outfile.write('Time (s)\t' + '\t'.join(str(item) 
                                               for item in CSDMG_Nodes))
        for i, j in enumerate(CSDMG_List):
            outfile.write('\n')
            tempList = [time[i]]
            for k in list(j):
                tempList.append(k)
            outfile.write('\t'.join(str(item) for item in tempList))
        outfile.close()
        print("\nDone!")
        print("\nThe output file will be named '{}".format(filename) + "'")
        print("\nIt will be in the same working directory as your Abaqus" + 
              " model\n")
    return

# Run the function
data_extract(jobName)