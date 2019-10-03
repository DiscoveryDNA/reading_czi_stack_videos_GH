#########################################################################################
############  190603_NormalizedIntensityGraphs.py                            ############
############ input: Directory with Czi files, intensity files                ############
############ normalized by sig                                               ############
############ output: signal around centers                                   ############
#########################################################################################
#! bin/python

import czifile
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation, feature, img_as_ubyte, morphology
import numpy as np
import os
from math import sqrt
import seaborn as sea
import pickle
import pandas as pd
from HelperFunctionsforImageProcessing import NucleiSegmentation_MIP_Dapi_Lamina, CZIMetadatatoDictionaries, CellEdgeMask, Threshold_DNASpots_triangle, Better_Threshold_DNASpots_triangle
from os import listdir
from optparse import OptionParser
from xml.etree import ElementTree as ET
import re



################ passing the arguments - this command should be given a directory of files that are to be analyzed in the same batch

def parse_options():
	parser = OptionParser()
	parser.add_option("-P", "--ProjectDirectory", dest="ProjectDirectory",
					  help="Project Directory")
	parser.add_option("-D", "--DataDirectory", dest="DataDirectory",
					  help="Directory with raw .czi files")
	parser.add_option("-R", "--NotebookResultsPath", dest="NotebookResultsPath",
					  help="NotebookResultsPath - path where output files should go")
	(options, args) = parser.parse_args()
	return options
            
options = parse_options()
parser = OptionParser()
if not options.ProjectDirectory:
    print("Project Directory option is missing\n")
    parser.print_help()
    exit(-1)
if not options.DataDirectory:
    print("Data Directory option is missing\n")
    parser.print_help()
    exit(-1)
if not options.NotebookResultsPath:
    print("NotebookResultsPath option is missing\n")
    parser.print_help()
    exit(-1)

#### Create the project directory, bin, notebookresultsPath

ProjectDirectory = str(options.ProjectDirectory) # the directory name as a string that is readable as a path
DataDirectory = str(options.DataDirectory) # Data Directory 
NotebookResultsName= str(options.NotebookResultsPath) #Results folder name

ProjectBin = ProjectDirectory + '/bin'
NotebookResultsPath = ProjectDirectory + '/results/' + NotebookResultsName + '/'
ProjectData = DataDirectory


assert os.path.exists(ProjectBin), "The ProjectBin directory , " + str(ProjectBin) + ' does not exist!' #if can't find the directory produces this error messaage

assert os.path.exists(NotebookResultsPath), "The results/output directory " +  str(NotebookResultsPath) + ' does not exist!' #if can't find the directory produces this error messaage

assert os.path.exists(ProjectData), "The project data directory " + str(ProjectData) + 'does not exist!' #if can't find the directory produces this error messaage

os.chdir(NotebookResultsPath)

#### Import the filelist
filelist = listdir(ProjectData) #list of file names
#print(filelist)

#### Add any file names that have .czi to a list target files
target_files = []
for fname in filelist:
    if fname.endswith('.czi'):
        target_files.append(fname)
    else:
         print("This file is not a CZI file - get outta here :", fname)
#print(target_files)



################ Make the empty picture for the summary pictures
### Sample 1
Sample1_647DotsinImage_647 = np.zeros((15,15))
Sample1_647DotsinImage_546 = np.zeros((15,15))
Sample1_647DotsinImage_488 = np.zeros((15,15))
Sample1_647DotsinImage_405 = np.zeros((15,15))

################ Make the empty picture for the summary pictures
### Sample 1
Sample1_546DotsinImage_647 = np.zeros((15,15))
Sample1_546DotsinImage_546 = np.zeros((15,15))
Sample1_546DotsinImage_488 = np.zeros((15,15))
Sample1_546DotsinImage_405 = np.zeros((15,15))

################ Make the empty Lists for the percent difference
PercentDiffList = []
PercentDiffList_Firsttwo = []

################ Initializing counts
AllCells = 0
ImageCount = 0

################ Make results file list for checking for pickled blob logs
resultsfilelist = listdir(NotebookResultsPath)

################ Initialized output file

OutputFile = open('SampleInfo.txt', 'w')
Fields = ['CziName', 'TotalCells', 'Total647Spots', 'Total546Spots', 'Center647_647_max','Center546_647_max', 'Center647_546_max','Center546_546_max', 'Center647_488_max','Center546_488_max', 'Center647_405_max', 'Center546_405_max', 'PercentDiff_Intensityof405', 'PercentDiff_Intensityof488', 'PercentDiff_Intensityof647', 'PercentDiff_Intensityof546']
Separator = '\t'
Separator.join(Fields)
OutputFile.write(str(Separator.join(Fields) + '\n'))

for CziName in target_files:

    #Function that takes the CZI file and splits it up into stacks based on the metadata
    AF405Stack, AF488Stack, AF647Stack, AF546Stack = CZIMetadatatoDictionaries(ProjectData, CziName) 
    
    ############ Threshold nuclei based on 405 - lamina and 488 - Protein ############  
    Watershed_ClearBorders = NucleiSegmentation_MIP_Dapi_Lamina(CziName, AF405Stack)
    TotalCells = len(np.unique(Watershed_ClearBorders)) - 1 
    print('Total Cells :' , TotalCells)

    ############ Mask and threshold DNA Fish image(s). ############
    ## Using Threshold_DNASpots_triangle because rank filter doesn't work on stacks
    AF647Stack_8int_gaus_Triangle, AF647Stack_8int_gaus_Triangle_labeled_array, num_features_647, AF546Stack_8int_gaus_Triangle, AF546Stack_8int_gaus_Triangle_labeled_array, num_features_546 = Threshold_DNASpots_triangle(CziName, AF647Stack, AF546Stack, Watershed_ClearBorders)


    ############ Check for previously pickeled blob log 
    blobs_log_647_fileName = str(CziName + "_blobs_log_647_file.txt")
    blobs_log_546_fileName = str(CziName + "_blobs_log_546_file.txt")

    if blobs_log_647_fileName in resultsfilelist:
        blobs_log_647_500 = pickle.load(open(str(blobs_log_647_fileName), "rb"))
        print(str(CziName + " 647channel has a blob log - importing that now"))
        #### Save bloblog_647Size for output file
        bloblog_647_size = str(np.size(blobs_log_647_500,0))
    else:
        ############ Blob Log on the stack
        blobs_log_647 = feature.blob_log(AF647Stack_8int_gaus_Triangle, min_sigma = 2)
        print(str("Amount of 647 Blobs detected: " + str(np.size(blobs_log_647,0))))
        
        ############ plot for each Z slice the blobs that it is circling
        blobs_log_647[:, 3] = blobs_log_647[:, 3] * sqrt(3) #takes everything in the third row (sigma) and multiplies that by the square root of 3

        ############ Randomly shuffle the blob log and pull out the first 500
        np.random.shuffle(blobs_log_647)
        blobs_log_647_500 = blobs_log_647[0:500,...]
        
        #### Pickle the blob logs to save them.
        with open(blobs_log_647_fileName, 'wb') as blobs_log_647_file:
            pickle.dump(blobs_log_647_500, blobs_log_647_file)

        #### Save bloblog_647Size for output file
        bloblog_647_size = str(np.size(blobs_log_647_500,0))
    if blobs_log_546_fileName in resultsfilelist:
        blobs_log_546_500 = pickle.load( open(str(blobs_log_546_fileName), "rb" ) )
        print(str(CziName + " 546 channel has a blob log - importing that now"))
        bloblog_546_size = str(np.size(blobs_log_546_500,0))
    else:
        ############ Blob Log on the stack
        blobs_log_546 = feature.blob_log(AF546Stack_8int_gaus_Triangle, min_sigma = 2)
        print(str("Amount of 546 Blobs detected: " + str(np.size(blobs_log_546,0))))
        ############ plot for each Z slice the blobs that it is circling
        blobs_log_546[:, 3] = blobs_log_546[:, 3] * sqrt(3)

        ############ Randomly shuffle the blob log and pull out the first 500
        np.random.shuffle(blobs_log_546)
        blobs_log_546_500 = blobs_log_546[0:500,...]

        #### Pickle the blob logs to save them.
        with open(blobs_log_546_fileName, 'wb') as blobs_log_546_file:
            pickle.dump(blobs_log_546_500, blobs_log_546_file)
        
        bloblog_546_size = str(np.size(blobs_log_546_500,0))

    ############ If the length of the list isn't 500 (less than that for some reason) Skip it because that means there are less blobs than that so its not a quality image
    if len(blobs_log_647_500) != 500:
        print(str(CziName + ' AF647 channel does not have enough spots'))
        continue
        
    if len(blobs_log_546_500) != 500:
        print(str(CziName + ' AF546 channel does not have enough spots'))
        continue
        
    ############ Make an average image for all 647 dots in the image
    Center647_647 = np.zeros((15,15))
    Center647_546 = np.zeros((15,15))
    Center647_488 = np.zeros((15,15))
    Center647_405 = np.zeros((15,15))
    
    for SliceNum in range(AF488Stack.shape[0]): #for each slice between 0 and the max number of slices
        for blob in blobs_log_647_500: #for each blob in the 647 blob log
            if blob[0] == SliceNum: #if the blob center is equal to the slice number.
                x = int(blob[2]) #The x value of the center is the third field of the bloblog output
                y = int(blob[1]) #The y value of the center is the second field of the bloblog output

                im_405 = AF405Stack[SliceNum] #grab out the slice 
                im_647 = AF647Stack[SliceNum]
                im_546 = AF546Stack[SliceNum]
                im_488 = AF488Stack[SliceNum]

                try:
                    Intensityof405 = im_405[y-7:y+8,x-7:x+8] #15 pixel cube
                    Center647_405 += Intensityof405 # is really just the sum of all the signal 

                    Intensityof647 = im_647[y-7:y+8,x-7:x+8]
                    Center647_647 += Intensityof647

                    Intensityof546 = im_546[y-7:y+8,x-7:x+8]
                    Center647_546 += Intensityof546

                    Intensityof488 = im_488[y-7:y+8,x-7:x+8]
                    Center647_488 += Intensityof488

                except ValueError: # Takes out anything that basically is less than 20+- in X/Y direction
                    #print ("ValueError", SliceNum,x,y) #to get around this I could just make it explicit in the statement that as long as the X is in a certain range
                    pass
                except IndexError:
                    #print ("IndexError", SliceNum,x,y)
                    pass

    Center647_647_max = np.amax(Center647_647)
    Center647_546_max = np.amax(Center647_546)
    Center647_488_max = np.amax(Center647_488)
    Center647_405_max = np.amax(Center647_405)

    Center546_647 = np.zeros((15,15))
    Center546_546 = np.zeros((15,15))
    Center546_488 = np.zeros((15,15))
    Center546_405 = np.zeros((15,15))

    for SliceNum in range(AF488Stack.shape[0]): #for each slice between 0 and the max number of slices
        for blob in blobs_log_546_500: #for each blob in the 647 blob log
            if blob[0] == SliceNum: #if the blob center is equal to the slice number.
                x = int(blob[2]) #The x value of the center is the third field of the bloblog output
                y = int(blob[1]) #The y value of the center is the second field of the bloblog output

                im_405 = AF405Stack[SliceNum] #grab out the slice 
                im_647 = AF647Stack[SliceNum]
                im_546 = AF546Stack[SliceNum]
                im_488 = AF488Stack[SliceNum]

                try:
                    Intensityof405 = im_405[y-7:y+8,x-7:x+8] #15 pixel cube
                    Center546_405 += Intensityof405 # is really just the sum of all the signal 

                    Intensityof647 = im_647[y-7:y+8,x-7:x+8]
                    Center546_647 += Intensityof647

                    Intensityof546 = im_546[y-7:y+8,x-7:x+8]
                    Center546_546 += Intensityof546

                    Intensityof488 = im_488[y-7:y+8,x-7:x+8]
                    Center546_488 += Intensityof488

                except ValueError: # Takes out anything that basically is less than 20+- in X/Y direction
                    #print ("ValueError", SliceNum,x,y) #to get around this I could just make it explicit in the statement that as long as the X is in a certain range
                    pass
                except IndexError:
                    #print ("IndexError", SliceNum,x,y)
                    pass

    Center546_647_max = np.amax(Center546_647)
    Center546_546_max = np.amax(Center546_546)
    Center546_488_max = np.amax(Center546_488)
    Center546_405_max = np.amax(Center546_405)

    print("Center647_647_max", Center647_647_max, Center546_647_max)
    print("Center647_546_max", Center647_546_max, Center546_546_max)
    print("Center647_488_max", Center647_488_max, Center546_488_max)
    print("Center647_405_max", Center647_405_max, Center546_405_max)

    MAX647 = max(Center647_647_max, Center546_647_max)
    MAX546 = max(Center647_546_max, Center546_546_max)
    MAX488 = max(Center647_488_max, Center546_488_max)
    MAX405 = max(Center647_405_max, Center546_405_max)
    
    ## Add the Czi image to the bigger image of all the sample 1 dots together (500 from each image)
    Sample1_647DotsinImage_647 += Center647_647
    Sample1_647DotsinImage_546 += Center647_546
    Sample1_647DotsinImage_488 += Center647_488
    Sample1_647DotsinImage_405 += Center647_405

    Sample1_546DotsinImage_647 += Center546_647
    Sample1_546DotsinImage_546 += Center546_546
    Sample1_546DotsinImage_488 += Center546_488
    Sample1_546DotsinImage_405 += Center546_405

    plt.figure
    ax = plt.subplots(figsize=(20, 10))
    plt.subplot(4,2,1)
    plt.title('Center647_647')
    plt.imshow(Center647_647, interpolation = 'none', vmax = MAX647)
    plt.subplot(4,2,2)
    plt.title('Center546_647')
    plt.imshow(Center546_647, interpolation = 'none', vmax = MAX647)
    plt.subplot(4,2,3)
    plt.title('Center647_546')
    plt.imshow(Center647_546, interpolation = 'none', vmax = MAX546)
    plt.subplot(4,2,4)
    plt.title('Center546_546')
    plt.imshow(Center546_546, interpolation = 'none', vmax = MAX546)
    plt.subplot(4,2,5)
    plt.title('Center647_488')
    plt.imshow(Center647_488, interpolation = 'none', vmax = MAX488)
    plt.subplot(4,2,6)
    plt.title('Center546_488')
    plt.imshow(Center546_488, interpolation = 'none', vmax = MAX488)
    plt.subplot(4,2,7)
    plt.title('Center647_405')
    plt.imshow(Center647_405, interpolation = 'none', vmax = MAX405)
    plt.subplot(4,2,8)
    plt.title('Center546_405')
    plt.imshow(Center546_405, interpolation = 'none', vmax = MAX405)
    plt.savefig(str(CziName) + "_SumofSignal_100.png")
    plt.close()
    
    
    ## Take the Fl intensity at each blob center and make histograms, calculate the percent difference between the neg and positive spots and then add that to the greater list to make a boxplot showing the range across different images
    AF647Sig_in647center = []
    AF546Sig_in647center = []
    AF488Sig_in647center = []
    AF405Sig_in647center = []

    for SliceNum in range(AF488Stack.shape[0]): #for each slice between 0 and the max number of slices
        for blob in blobs_log_647_500: #for each blob in the 647 blob log
            if blob[0] == SliceNum: #if the blob center is equal to the slice number.
                x = int(blob[2]) #The x value of the center is the third field of the bloblog output
                y = int(blob[1]) #The y value of the center is the second field of the bloblog output

                im_405 = AF405Stack[SliceNum] #grab out the slice 
                im_647 = AF647Stack[SliceNum]
                im_546 = AF546Stack[SliceNum]
                im_488 = AF488Stack[SliceNum]

                Intensityof405 = np.zeros((15,15))
                Intensityof647 = np.zeros((15,15))
                Intensityof546 = np.zeros((15,15))
                Intensityof488 = np.zeros((15,15))

                try:
                    Intensityof405 = im_405[y,x] #1 pixel 
                    AF405Sig_in647center.append(Intensityof405) # is really just the sum of all the signal 

                    Intensityof647 = im_647[y,x]
                    AF647Sig_in647center.append(Intensityof647)

                    Intensityof546 = im_546[y,x]
                    AF546Sig_in647center.append(Intensityof546)

                    Intensityof488 = im_488[y,x]
                    AF488Sig_in647center.append(Intensityof488)

                except ValueError: # Takes out anything that basically is less than 20+- in X/Y direction
                    #print ("ValueError", SliceNum,x,y) #to get around this I could just make it explicit in the statement that as long as the X is in a certain range
                    pass
                except IndexError:
                    #print ("IndexError", SliceNum,x,y)
                    pass


    AF647Sig_in546center = []
    AF546Sig_in546center = []
    AF488Sig_in546center = []
    AF405Sig_in546center = []

    for SliceNum in range(AF488Stack.shape[0]): #for each slice between 0 and the max number of slices
        for blob in blobs_log_546_500: #for each blob in the 647 blob log
            if blob[0] == SliceNum: #if the blob center is equal to the slice number.
                x = int(blob[2]) #The x value of the center is the third field of the bloblog output
                y = int(blob[1]) #The y value of the center is the second field of the bloblog output

                im_405 = AF405Stack[SliceNum] #grab out the slice 
                im_647 = AF647Stack[SliceNum]
                im_546 = AF546Stack[SliceNum]
                im_488 = AF488Stack[SliceNum]

                Intensityof405 = np.zeros((15,15))
                Intensityof647 = np.zeros((15,15))
                Intensityof546 = np.zeros((15,15))
                Intensityof488 = np.zeros((15,15))

                try:
                    Intensityof405 = im_405[y,x] #15 pixel cube
                    AF405Sig_in546center.append(Intensityof405)# is really just the sum of all the signal 

                    Intensityof647 = im_647[y,x]
                    AF647Sig_in546center.append(Intensityof647)

                    Intensityof546 = im_546[y,x]
                    AF546Sig_in546center.append(Intensityof546)

                    Intensityof488 = im_488[y,x]
                    AF488Sig_in546center.append(Intensityof488)

                except ValueError: # Takes out anything that basically is less than 20+- in X/Y direction
                    #print ("ValueError", SliceNum,x,y) #to get around this I could just make it explicit in the statement that as long as the X is in a certain range
                    pass
                except IndexError:
                    #print ("IndexError", SliceNum,x,y)
                    pass

    ### Make scatterplots of Fluorescent intensity in the dot center.
    plt.figure()
    sea.distplot(AF488Sig_in647center, color=sea.xkcd_rgb["deep purple"])
    sea.distplot(AF488Sig_in546center, color=sea.xkcd_rgb["slate blue"])
    plt.savefig(str(CziName) + "AF488Sig_hist.png")
    plt.close()

    plt.figure()
    sea.distplot(AF647Sig_in647center, color=sea.xkcd_rgb["deep purple"])
    sea.distplot(AF647Sig_in546center, color=sea.xkcd_rgb["slate blue"])
    plt.savefig(str(CziName) + "AF647Sig_hist.png")
    plt.close()

    plt.figure()
    sea.distplot(AF546Sig_in647center, color=sea.xkcd_rgb["deep purple"])
    sea.distplot(AF546Sig_in546center, color=sea.xkcd_rgb["slate blue"])
    plt.savefig(str(CziName) + "AF546Sig_hist.png")
    plt.close()

    plt.figure()
    sea.distplot(AF405Sig_in647center, color=sea.xkcd_rgb["deep purple"])
    sea.distplot(AF405Sig_in546center, color=sea.xkcd_rgb["slate blue"])
    plt.savefig(str(CziName) + "AF405Sig_hist.png")
    plt.close()

    ## Sum the center intensity for each spot in image
    SumIntensityof405_647 = sum(AF405Sig_in647center)
    SumIntensityof488_647 = sum(AF488Sig_in647center)
    SumIntensityof647_647 = sum(AF647Sig_in647center)
    SumIntensityof546_647 = sum(AF546Sig_in647center)

    SumIntensityof405_546 = sum(AF405Sig_in546center)
    SumIntensityof488_546 = sum(AF488Sig_in546center)
    SumIntensityof647_546 = sum(AF647Sig_in546center)
    SumIntensityof546_546 = sum(AF546Sig_in546center)

    ## Calculate the Percent difference of the fluorescent intensity at each center for the positive and negative spots
    PercentDiff_Intensityof405 = (SumIntensityof405_647/SumIntensityof405_546) * 100
    PercentDiff_Intensityof488 = (SumIntensityof488_647/SumIntensityof488_546) * 100
    PercentDiff_Intensityof647 = (SumIntensityof647_647/SumIntensityof647_546) * 100
    PercentDiff_Intensityof546 = (SumIntensityof546_647/SumIntensityof546_546) * 100

    print("PercentDiff_Intensityof405 : ", PercentDiff_Intensityof405)
    print("PercentDiff_Intensityof488 : ", PercentDiff_Intensityof488)
    print("PercentDiff_Intensityof647 : ", PercentDiff_Intensityof647)
    print("PercentDiff_Intensityof546 : ", PercentDiff_Intensityof546)

    ### Add that to the list of the percentages for each image

    PercentDiffSampleList = [PercentDiff_Intensityof488, PercentDiff_Intensityof405, PercentDiff_Intensityof647, PercentDiff_Intensityof546]
    PercentDiffList.append(PercentDiffSampleList)
    
    PercentDiffSampleList_Firsttwo = [PercentDiff_Intensityof488, PercentDiff_Intensityof405]
    PercentDiffList_Firsttwo.append(PercentDiffSampleList_Firsttwo)
    
    ## Delete the blob logs from memory
    if 'blobs_log_647' in locals():
        del blobs_log_647
    if 'blobs_log_546' in locals():
        del blobs_log_546

    #### Output file write
    #Fields = ['CziName', 'TotalCells', 'Total647Spots', 'Total546Spots', 'Center647_647_max','Center546_647_max', 'Center647_546_max','Center546_546_max', 'Center647_488_max','Center546_488_max', 'Center647_405_max', 'Center546_405_max', 
    # ###'PercentDiff_Intensityof405', 'PercentDiff_Intensityof488', 'PercentDiff_Intensityof647', 'PercentDiff_Intensityof546']
    # print("PercentDiff_Intensityof405 : ", PercentDiff_Intensityof405)
    # print("PercentDiff_Intensityof488 : ", PercentDiff_Intensityof488)
    # print("PercentDiff_Intensityof647 : ", PercentDiff_Intensityof647)
    # print("PercentDiff_Intensityof546 : ", PercentDiff_Intensityof546)
    outputValuesList = [str(CziName), str(TotalCells),  str(bloblog_647_size), str(bloblog_546_size), str(Center647_647_max), str(Center546_647_max), str(Center647_546_max), str(Center546_546_max), str(Center647_488_max), str(Center546_488_max), str(Center647_405_max), str(Center546_405_max), str(PercentDiff_Intensityof405), str(PercentDiff_Intensityof488), str(PercentDiff_Intensityof647), str(PercentDiff_Intensityof546)]
    #Separator.join(outputValuesList) 
    OutputFile.write(str(Separator.join(outputValuesList) + '\n'))
    AllCells += TotalCells
    ImageCount += 1

######## Make the 647 summary figure for each sample
S1_Center647_647_max = np.amax(Sample1_647DotsinImage_647)
S1_Center647_546_max = np.amax(Sample1_647DotsinImage_546)
S1_Center647_488_max = np.amax(Sample1_647DotsinImage_488)
S1_Center647_405_max = np.amax(Sample1_647DotsinImage_405)

S1_Center546_647_max = np.amax(Sample1_546DotsinImage_647)
S1_Center546_546_max = np.amax(Sample1_546DotsinImage_546)
S1_Center546_488_max = np.amax(Sample1_546DotsinImage_488)
S1_Center546_405_max = np.amax(Sample1_546DotsinImage_405)

print("Center647_647_max", S1_Center647_647_max, S1_Center546_647_max)
print("Center647_546_max", S1_Center647_546_max, S1_Center546_546_max)
print("Center647_488_max", S1_Center647_488_max, S1_Center546_488_max)
print("Center647_405_max", S1_Center647_405_max, S1_Center546_405_max)

S1MAX647 = max(S1_Center647_647_max, S1_Center546_647_max)
S1MAX546 = max(S1_Center647_546_max, S1_Center546_546_max)
S1MAX488 = max(S1_Center647_488_max, S1_Center546_488_max)
S1MAX405 = max(S1_Center647_405_max, S1_Center546_405_max)

#### Output file write

CompositeValuesList = ['CompositeSample', str(AllCells),  str(500 * ImageCount), str(500 * ImageCount), str(S1_Center647_647_max), str(S1_Center546_647_max), str(S1_Center647_546_max), str(S1_Center546_546_max), str(S1_Center647_488_max), str(S1_Center546_488_max), str(S1_Center647_405_max), str(S1_Center546_405_max)]
OutputFile.write(str(Separator.join(CompositeValuesList) + '\n'))

#### Sample 1

plt.figure
ax = plt.subplots(figsize=(20, 10))
plt.subplot(4,2,1)
plt.title('Sample1_647DotsinImage_647')
plt.imshow(Sample1_647DotsinImage_647, interpolation = 'none', vmax = S1MAX647)
plt.subplot(4,2,2)
plt.title('Sample1_546DotsinImage_647')
plt.imshow(Sample1_546DotsinImage_647, interpolation = 'none', vmax = S1MAX647)
plt.subplot(4,2,3)
plt.title('Sample1_647DotsinImage_546')
plt.imshow(Sample1_647DotsinImage_546, interpolation = 'none', vmax = S1MAX546)
plt.subplot(4,2,4)
plt.title('Sample1_546DotsinImage_546')
plt.imshow(Sample1_546DotsinImage_546, interpolation = 'none', vmax = S1MAX546)
plt.subplot(4,2,5)
plt.title('Sample1_647DotsinImage_488')
plt.imshow(Sample1_647DotsinImage_488, interpolation = 'none', vmax = S1MAX488)
plt.subplot(4,2,6)
plt.title('Sample1_546DotsinImage_488')
plt.imshow(Sample1_546DotsinImage_488, interpolation = 'none', vmax = S1MAX488)
plt.subplot(4,2,7)
plt.title('Sample1_647DotsinImage_405')
plt.imshow(Sample1_647DotsinImage_405, interpolation = 'none', vmax = S1MAX405)
plt.subplot(4,2,8)
plt.title('Sample1_546DotsinImage_405')
plt.imshow(Sample1_546DotsinImage_405, interpolation = 'none', vmax = S1MAX405)
plt.savefig("Sample1_" + str(CziName) + "_SumofSignal_100.png")
plt.close()


#### Make the boxplots with all of the differences between negative and positive regions
#Make list of lists
#PercentDiffDataFrameList = [PercentDiff_Intensityof488_List, PercentDiff_Intensityof405_List, PercentDiff_Intensityof647_List, PercentDiff_Intensityof546_List]

Labels = ['AF488', 'AF405', 'AF647', 'AF546']

PercentDiffDataFrame = pd.DataFrame.from_records(PercentDiffList, columns = Labels)

plt.figure()
sea.boxplot(data = PercentDiffDataFrame, color=sea.xkcd_rgb["slate blue"])
plt.savefig("190316_PercDiff_Sample1_Histogram.png")
plt.close()

Labels2 = ['AF488', 'AF405']
PercentDiffDataFrameFirst2 = pd.DataFrame.from_records(PercentDiffList_Firsttwo, columns = Labels2)

plt.figure()
sea.boxplot(data = PercentDiffDataFrameFirst2, color=sea.xkcd_rgb["slate blue"])
plt.savefig("190316_PercDiff_Sample1_Histogram_Just488_405.png")
plt.close()
