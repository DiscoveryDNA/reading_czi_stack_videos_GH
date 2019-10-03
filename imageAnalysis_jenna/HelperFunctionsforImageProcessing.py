
########################################################################################
############  HelperFunctionsforImageProcessing.py              ############
############ input: Czi stack                                               ############
############ output: .txt file with how many spots per nucleus              ############
########################################################################################
#! bin/python

import sys
import czifile
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation, transform, exposure, img_as_ubyte, feature, morphology
from skimage.morphology import disk, ball
import numpy as np
from os import listdir
import os
from xml.etree import ElementTree as ET
import re
import os.path
from os import path
import tifffile



def CellEdgeMask(combined_405_488_watershed_image_noborders, iterations):
    CellEdge = np.zeros_like(combined_405_488_watershed_image_noborders) #make an array the same size and data type as the watershed but filled with zeros
    for cell in np.unique(combined_405_488_watershed_image_noborders):
        cell_mask = combined_405_488_watershed_image_noborders==cell
        ErodedCellMask = ndi.binary_erosion(cell_mask, iterations=iterations) # Increase iterations to make boundary wider!
        edge_mask = np.logical_xor(cell_mask, ErodedCellMask) #  Create the cell edge mask
        CellEdge[edge_mask] = cell
    CellEdge_mask = np.ma.array(CellEdge, mask=CellEdge==0)
    return(CellEdge_mask)

def ExperimentInformation(ExperimentDesignFilePath):
    ExperimentDesignFile = open(str(ExperimentDesignFilePath))
    ExperimentDesignlines = ExperimentDesignFile.readlines()
    ExperimentDesignlines.pop(0)
    
    AF647Dict = {}
    AF546Dict = {}
    AF488Dict = {}
    AF405Dict = {}
    
    for line in ExperimentDesignlines:
        line = line.strip()
        field = line.split('\t')
        Date = field[0]
        ExpName = field[1]
        SampleNum = field[2]
        AF647 = field[3]
        AF546 = field[4]
        AF488 = field[5]
        AF405 = field[6]
        AF647Dict[SampleNum] = AF647
        AF546Dict[SampleNum] = AF546
        AF488Dict[SampleNum] = AF488
        AF405Dict[SampleNum] = AF405
        
    return(AF647Dict, AF546Dict, AF488Dict, AF405Dict)

def segment_nuclei_405_488(CziName, AF405Stack, AF488Stack):
    
    ############ Max intensity projection of lamina and 488 staining
    AF405_MIP = np.amax(AF405Stack, axis = 0) 
    AF488_MIP = np.amax(AF488Stack, axis = 0)#make a max intensity projection of the z axis 
    #plt.imshow(AF405_MIP)
    #plt.show()

    ############ Turn into 8 bit for masking
    AF405_MIP_8int = img_as_ubyte(AF405_MIP)
    AF488_MIP_8int = img_as_ubyte(AF488_MIP)
    # plt.imshow(AF405_MIP_8int)
    # plt.show()

    ############ GAUSSIAN FILTER to determine background
    
    combined = AF405_MIP_8int + AF488_MIP_8int

    combined_GausFilt = ndi.filters.gaussian_filter(combined, 20)

    ########### ADAPTIVE THRESHOLDING - combined
    localthresh = filters.threshold_local(combined_GausFilt, 75)
    combined_localtheshold = combined_GausFilt > localthresh

    ########### ADAPTIVE THRESHOLDING - 488 
    AF488_MIP_8int_GausFilt = ndi.filters.gaussian_filter(AF488_MIP_8int, 10)

    ########### WATERSHED to break nuclei apart
    
    #### Distance transform
    distancetransform_combined_final = ndi.distance_transform_edt(combined_localtheshold)


    #### smoothen the distance transform
    thresh = 10
    distancetransform_combined_final_gaus = ndi.filters.gaussian_filter(distancetransform_combined_final, thresh)
   
    #### Retrieve the local maxima from the distance transform
    Local_max = feature.peak_local_max(distancetransform_combined_final_gaus, indices = False, min_distance = 20)
    Local_max_bigger = ndi.filters.maximum_filter(Local_max, size=5)
    Local_max_mask = np.ma.array(Local_max_bigger, mask=Local_max_bigger==0) #makes a mask so that I can visualize on top of the original image

    #### label each center
    labeled_array_segmentation, num_features_seg = ndi.label(Local_max_mask)
    labeled_array_segmentation_mask = np.ma.array(labeled_array_segmentation, mask=labeled_array_segmentation==0) #make a segmentation mask
    watershed_image = morphology.watershed(~AF488_MIP_8int_GausFilt, labeled_array_segmentation_mask, mask=combined_localtheshold)
   
    ########### REMOVE BORDER CELLS
    combined_405_488_watershed_image_noborders = segmentation.clear_border(watershed_image)

    ########### PLOT OUTPUT OF SEGMENTATION FOR EACH IMAGE
    
    ### Make a random colormap
    vals = np.linspace(0,1,256)
    np.random.shuffle(vals)
    cmap = plt.cm.colors.ListedColormap(plt.cm.jet(vals))
    
    
    plt.figure
    ax = plt.subplots(figsize=(20, 10))
    plt.subplot(2,2,1)
    plt.title('Input 488 + 405 combined image - ' + str(CziName))
    plt.imshow(combined)
    plt.subplot(2,2,2)
    plt.title('Input 488 + 405 combined image gaussian filtered - ' + str(CziName))
    plt.imshow(combined_GausFilt)
    plt.subplot(2,2,3)
    plt.title('Segmented watershed Image - ' + str(CziName))
    plt.imshow(watershed_image, cmap = cmap)
    plt.imshow(AF488_MIP_8int_GausFilt, alpha = 0.3, cmap = "gray")
    plt.subplot(2,2,4)
    plt.title('Final segmented cells used in analysis - ' + str(CziName))
    plt.imshow(combined_405_488_watershed_image_noborders)
    plt.savefig('Segmentation_results' + str(CziName) + '.png')
    plt.close()
    print("Printed PDF of segmentation results for ", CziName)
    print("segmented ", num_features_seg, "cells in ", CziName, ".")
    return combined_405_488_watershed_image_noborders

def Threshold_DNASpots_triangle(CziName, AF647Stack, AF546Stack, combined_405_488_watershed_image_noborders):
    
    ############ Convert images into 8bit for thresholding
    AF647Stack_8int = img_as_ubyte(AF647Stack)
    AF546Stack_8int = img_as_ubyte(AF546Stack)

    ############ Gaussian Filter 
    AF647Stack_8int_gaus = ndi.filters.gaussian_filter(AF647Stack_8int, 1)
    AF546Stack_8int_gaus = ndi.filters.gaussian_filter(AF546Stack_8int, 1)

    ############ Threshold using Triangle method
    AF647Stack_8int_gaus_Triangle_thresh = filters.threshold_triangle(AF647Stack_8int_gaus) +2
    AF647Stack_8int_gaus_Triangle = AF647Stack_8int_gaus > AF647Stack_8int_gaus_Triangle_thresh
    
    AF546Stack_8int_gaus_Triangle_thresh = filters.threshold_triangle(AF546Stack_8int_gaus) +2
    AF546Stack_8int_gaus_Triangle = AF546Stack_8int_gaus > AF546Stack_8int_gaus_Triangle_thresh

    ############ Itemize the spots
    AF647Stack_8int_gaus_Triangle_labeled_array, num_features_647 = ndi.label(AF647Stack_8int_gaus_Triangle)
    AF546Stack_8int_gaus_Triangle_labeled_array, num_features_546 = ndi.label(AF546Stack_8int_gaus_Triangle)

    print("Triangle Thresholded Images complete for ", CziName)
    print("Found ", num_features_647, " spots in the 647 channel.")
    print("Found ", num_features_546, " spots in the 546 channel.")

    #### Creating the Nuclei edges
    CellEdge = np.zeros_like(combined_405_488_watershed_image_noborders) #make an array the same size and data type as the watershed but filled with zeros
    for cell in np.unique(combined_405_488_watershed_image_noborders):
        cell_mask = combined_405_488_watershed_image_noborders==cell
        ErodedCellMask = ndi.binary_erosion(cell_mask, iterations=1) # Increase iterations to make boundary wider!
        edge_mask = np.logical_xor(cell_mask, ErodedCellMask) #  Create the cell edge mask
        CellEdge[edge_mask] = cell
    CellEdge_mask = np.ma.array(CellEdge, mask=CellEdge==0)

    fig = plt.figure()
    fig.suptitle('Stack Triangle + Thresholding Output' + str(CziName))
    ax = plt.subplots(figsize=(20, 10))
    plt.subplot(2,2,1)
    plt.title('AF647Stack_8int_gaus')
    plt.imshow(np.amax(AF647Stack_8int_gaus, axis = 0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,2,2)
    plt.title('AF647Stack_8int_gaus_Triangle')
    plt.imshow(np.amax(AF647Stack_8int_gaus_Triangle, axis = 0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,2,3)
    plt.title('AF546Stack_8int_gaus')
    plt.imshow(np.amax(AF546Stack_8int_gaus, axis = 0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,2,4)
    plt.title('AF546Stack_8int_gaus_Triangle')
    plt.imshow(np.amax(AF546Stack_8int_gaus_Triangle, axis = 0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.savefig('DNASpot_Filter_Thresholding' + str(CziName) + '.png')
    plt.close()

    return AF647Stack_8int_gaus_Triangle, AF647Stack_8int_gaus_Triangle_labeled_array, num_features_647, AF546Stack_8int_gaus_Triangle, AF546Stack_8int_gaus_Triangle_labeled_array, num_features_546

def MIP_Threshold_DNASpots_triangle_BilateralRankMean(CziName, AF647Stack, AF546Stack, combined_405_488_watershed_image_noborders):
   
   ###### Turn into 8 bit
    AF647Stack_8int = img_as_ubyte(AF647Stack)
    AF546Stack_8int = img_as_ubyte(AF546Stack)

    ###### Small Gaussian Filter
    AF647Stack_8int_gaus = ndi.filters.gaussian_filter(AF647Stack_8int, 1)
    AF546Stack_8int_gaus = ndi.filters.gaussian_filter(AF546Stack_8int, 1)

    ###### MIP
    AF647Stack_8int_gaus_MIP = np.amax(AF647Stack_8int_gaus, axis = 0)
    AF546Stack_8int_gaus_MIP = np.amax(AF546Stack_8int_gaus, axis = 0)

    ###### Rank Mean bilateral filter
    i = 5
    struct = (np.mgrid[:i,:i][0] - np.floor(i/2))**2 + (np.mgrid[:i,:i][1] - np.floor(i/2))**2 <= np.floor(i/2)**2
    AF647Stack_8int_gaus_MIP_rank = filters.rank.mean_bilateral(AF647Stack_8int_gaus_MIP, struct)
    AF546Stack_8int_gaus_MIP_rank = filters.rank.mean_bilateral(AF546Stack_8int_gaus_MIP, struct)

    ###### Triangle Threshold
    AF647Stack_8int_gaus_MIP_rank_Triangle_thresh = filters.threshold_triangle(AF647Stack_8int_gaus_MIP_rank)
    AF647Stack_8int_gaus_MIP_rank_Triangle = AF647Stack_8int_gaus_MIP_rank > AF647Stack_8int_gaus_MIP_rank_Triangle_thresh

    AF546Stack_8int_gaus_MIP_rank_Triangle_thresh = filters.threshold_triangle(AF546Stack_8int_gaus_MIP_rank)
    AF546Stack_8int_gaus_MIP_rank_Triangle = AF546Stack_8int_gaus_MIP_rank > AF546Stack_8int_gaus_MIP_rank_Triangle_thresh

    ###### Watershed
    AF647distancetransform = ndi.distance_transform_edt(AF647Stack_8int_gaus_MIP_rank_Triangle)
    AF546distancetransform = ndi.distance_transform_edt(AF546Stack_8int_gaus_MIP_rank_Triangle)

    AF647Local_max = feature.peak_local_max(AF647distancetransform, indices = False)
    AF546Local_max = feature.peak_local_max(AF546distancetransform, indices = False)

    AF647Local_max_mask = np.ma.array(AF647Local_max, mask=AF647Local_max==0) #makes a mask so that I can visualize on top of the original image
    AF546Local_max_mask = np.ma.array(AF546Local_max, mask=AF546Local_max==0)

    AF647Local_max_mask_array, AF647Local_max_mask_array_num_features = ndi.label(AF647Local_max_mask)
    AF546Local_max_mask_array, AF546Local_max_mask_array_num_features = ndi.label(AF546Local_max_mask)

    AF647Local_array_mask = np.ma.array(AF647Local_max_mask_array, mask=AF647Local_max_mask_array==0) #make a segmentation mask
    AF546Local_array_mask = np.ma.array(AF546Local_max_mask_array, mask=AF546Local_max_mask_array==0)

    AF647watershed_image = morphology.watershed(AF647Stack_8int_gaus_MIP_rank_Triangle, AF647Local_array_mask)
    AF546watershed_image = morphology.watershed(AF546Stack_8int_gaus_MIP_rank_Triangle, AF546Local_array_mask)

    print("MIP Bilateral mean + Triangle Thresholded Images complete for ", CziName)
    print("Found ", AF647Local_max_mask_array_num_features, " spots in the 647 channel.")
    print("Found ", AF546Local_max_mask_array_num_features, " spots in the 546 channel.")
    print("Watershed Found ", len(np.unique(AF647watershed_image)), " spots in the 647 channel.")
    print("Watershed Found ", len(np.unique(AF546watershed_image)), " spots in the 546 channel.")

    #### Creating the Nuclei edges
    CellEdge = np.zeros_like(AF647Stack_8int_gaus_MIP) #make an array the same size and data type as the watershed but filled with zeros
    for cell in np.unique(combined_405_488_watershed_image_noborders):
        cell_mask = combined_405_488_watershed_image_noborders==cell
        ErodedCellMask = ndi.binary_erosion(cell_mask, iterations=1) # Increase iterations to make boundary wider!
        edge_mask = np.logical_xor(cell_mask, ErodedCellMask) #  Create the cell edge mask
        CellEdge[edge_mask] = cell
    CellEdge_mask = np.ma.array(CellEdge, mask=CellEdge==0)

    fig = plt.figure()
    fig.suptitle('MIP Bilateral mean + Thresholding Output' + str(CziName))
    ax = plt.subplots(figsize=(20, 10))
    plt.subplot(2,4,1)
    plt.title('AF647Stack_8int_gaus_MIP')
    plt.imshow(AF647Stack_8int_gaus_MIP)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,2)
    plt.title('AF647Stack_8int_gaus_MIP_rank')
    plt.imshow(AF647Stack_8int_gaus_MIP_rank)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,3)
    plt.title('AF647Stack_8int_gaus_MIP_rank_Triangle')
    plt.imshow(AF647Stack_8int_gaus_MIP_rank_Triangle)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,4)
    plt.title('AF647watershed_image')
    plt.imshow(AF647watershed_image)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,5)
    plt.title('AF546Stack_8int_gaus_MIP')
    plt.imshow(AF546Stack_8int_gaus_MIP)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,6)
    plt.title('AF546Stack_8int_gaus_MIP_rank')
    plt.imshow(AF546Stack_8int_gaus_MIP_rank)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,7)
    plt.title('AF546Stack_8int_gaus_MIP_rank_Triangle')
    plt.imshow(AF546Stack_8int_gaus_MIP_rank_Triangle)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,8)
    plt.title('AF546watershed_image')
    plt.imshow(AF546watershed_image)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.savefig('DNASpot_Filter_Thresholding' + str(CziName) + '.png')
    plt.close()

    return(AF647watershed_image, AF546watershed_image, AF647Stack_8int_gaus_MIP_rank_Triangle, AF546Stack_8int_gaus_MIP_rank_Triangle)

def CZIMetadatatoDictionaries(InputDirectory, CziName):
    czi = czifile.CziFile(InputDirectory + str(CziName))
    czi_array = czifile.imread(InputDirectory + str(CziName)) #read the czi file.
    czi_array = czi_array.squeeze() #take out the dimensions that are not important
    #print(czi_array.shape)
    
    ####### Extract the metadata
    metadata = czi.metadata     #reading the metadata from CZI
    root = ET.fromstring(metadata)  #loading metadata into XML object
    ##### Making a dictionry from all of the channel data only if it has ID that hs the channel number as the key and the dye name as the value
    ChannelDictionary = {}
    for neighbor in root.iter('Channel'):  
        TempDict = {}
        TempDict = neighbor.attrib
        if 'Id' in TempDict: #for the metadata lines that start with ID
            #print(TempDict) #test
            Search = r"(\w+):(\d)" #separate the channel:1 into two parts .. only keep the number
            Result = re.search(Search, TempDict['Id'])
            Search2 = r"(\w+)-(.+)"
            Result2 = re.search(Search2, TempDict['Name'])
            ChannelDictionary[Result2.group(1)] = Result.group(2) #make a new dictionary where that number (0 based!) is the channel/key to the and the value is the dye name
    #print(ChannelDictionary)

    ####### pull out the channels and make stacks
    if "AF405" in ChannelDictionary.keys():
        AF405index = ChannelDictionary["AF405"]
        AF405Stack = czi_array[int(AF405index),...]
    else:
        print("AF405 is not in this file")
        AF488Stack = 'empty'
    
    if "AF488" in ChannelDictionary.keys():
        AF488index = ChannelDictionary["AF488"]
        AF488Stack = czi_array[int(AF488index),...]

    else:
        print("AF488 is not in this file")
        AF488Stack = 'empty'
    
    if "AF647" in ChannelDictionary.keys():
        AF647index = ChannelDictionary["AF647"]
        AF647Stack = czi_array[int(AF647index),...]
    else:
        print("AF647 is not in this file")
        AF647Stack = 'empty'

    if "AF546" in ChannelDictionary.keys() :
        AF546index = ChannelDictionary["AF546"]
        AF546Stack = czi_array[int(AF546index),...]
    elif "At550" in ChannelDictionary.keys() :
        AF546index = ChannelDictionary["At550"]
        AF546Stack = czi_array[int(AF546index),...]
    else:
        print("AF546 is not in this file")
        AF546Stack = 'empty' 
    
    return(AF405Stack, AF488Stack, AF647Stack, AF546Stack)

def Segment_nuclei_405_only(CziName, AF405Stack):
    
    ############ Max intensity projection of lamina and 488 staining
    AF405_MIP = np.amax(AF405Stack, axis = 0)  
    #plt.imshow(AF405_MIP)
    #plt.show()

    ############ Turn into 8 bit for masking
    AF405_MIP_8int = img_as_ubyte(AF405_MIP)
    # plt.imshow(AF405_MIP_8int)
    # plt.show()

    ############ GAUSSIAN FILTER


    AF405_GausFilt = ndi.filters.gaussian_filter(AF405_MIP_8int, 1)

    ########### MEDIAN THRESHOLDING - Center points 
    try:
        thresh = filters.threshold_minimum(AF405_GausFilt)
    except RuntimeError:
        print(CziName, " used mean filter plus 5")
        thresh = filters.threshold_mean(AF405_GausFilt) - 5
    AF405_GausFilt_minthresh = AF405_GausFilt > thresh
    AF405_GausFilt_minthresh_remholes = morphology.remove_small_holes(AF405_GausFilt_minthresh, area_threshold=1000)

    ########### MEDIAN + 40 - edge points
    AF405_GausFilt_edges = AF405_GausFilt > (thresh + 40)
    AF405_GausFilt_edges_closed = ndi.binary_closing(AF405_GausFilt_edges,iterations= 10, structure=morphology.disk(1) )
    #plt.imshow(~AF405Stack_MIP_gausedges_closed)
    
    ########### MASK of center + edge points 
    Mask =  ~AF405_GausFilt_edges_closed * AF405_GausFilt_minthresh_remholes
    
    ########### WATERSHED to break nuclei apart
    #### Distance transform
    distancetransform_combined_final = ndi.distance_transform_edt(Mask)


    #### smoothen the distance transform
    distancetransform_combined_final_gaus = ndi.filters.gaussian_filter(distancetransform_combined_final, 10)
   
    #### Retrieve the local maxima from the distance transform
    Local_max = feature.peak_local_max(distancetransform_combined_final_gaus, indices = False, min_distance = 40)
    Local_max_bigger = ndi.filters.maximum_filter(Local_max, size=5)
    Local_max_mask = np.ma.array(Local_max_bigger, mask=Local_max_bigger==0) #makes a mask so that I can visualize on top of the original image

    #### label each center
    labeled_array_segmentation, num_features_seg = ndi.label(Local_max_mask)
    labeled_array_segmentation_mask = np.ma.array(labeled_array_segmentation, mask=labeled_array_segmentation==0) #make a segmentation mask
    watershed_image = morphology.watershed(Mask, labeled_array_segmentation_mask, mask=Mask)
   
    ########### REMOVE BORDER CELLS
    AF405_only_watershed_noborders = segmentation.clear_border(watershed_image)

    ########### PLOT OUTPUT OF SEGMENTATION FOR EACH IMAGE
    
    ### Make a random colormap
    vals = np.linspace(0,1,256)
    np.random.shuffle(vals)
    cmap = plt.cm.colors.ListedColormap(plt.cm.jet(vals))
    
    
    plt.figure
    ax = plt.subplots(figsize=(20, 10))
    plt.subplot(3,2,1)
    plt.title('Input 405  image - ' + str(CziName))
    plt.imshow(AF405_MIP_8int)
    plt.subplot(3,2,2)
    plt.title('Nuclei mask - post median thresholding - ' + str(CziName))
    plt.imshow(AF405_GausFilt_minthresh_remholes)
    plt.subplot(3,2,3)
    plt.title('Edges mask - ' + str(CziName))
    plt.imshow(AF405_GausFilt_edges_closed)
    plt.subplot(3,2,3)
    plt.title('Final mask - ' + str(CziName))
    plt.imshow(Mask)
    plt.subplot(3,2,4)
    plt.title('Segmented watershed Image - ' + str(CziName))
    plt.imshow(watershed_image, cmap = cmap)
    plt.imshow(AF405_MIP_8int, alpha = 0.4, cmap = "gray")
    plt.subplot(3,2,5)
    plt.title('Final segmented cells used in analysis - ' + str(CziName))
    plt.imshow(AF405_only_watershed_noborders)
    plt.savefig('Segmentation_results' + str(CziName) + '.png')
    plt.close()
    print("Printed PDF of segmentation results for ", CziName)
    print("segmented ", num_features_seg, "cells in ", CziName, ".")
    return(AF405_only_watershed_noborders)

def ImageList_4byX_Montage(ImageList, ImageTitleList, FigureTitle):
    ImageNumber = len(ImageList)
    Row_Num = int(ImageNumber / 4)
    fig, axs = plt.subplots(nrows = Row_Num , ncols = 4, figsize=(50, 50))
    for ax, image, title in zip(axs.flat, ImageList, ImageTitleList) : 
        ax.imshow(image)
        ax.set_title(str(title))
    plt.tight_layout()
    plt.savefig(str(FigureTitle))
def Stack_Threshold_DNASpots_triangle_BilateralRankMean(CziName, AF647Stack, AF546Stack, combined_405_488_watershed_image_noborders):
    ###### Turn into 8 bit
    AF647Stack_8int = img_as_ubyte(AF647Stack)
    AF546Stack_8int = img_as_ubyte(AF546Stack)

    ###### Small Gaussian Filter
    AF647Stack_8int_gaus = ndi.filters.gaussian_filter(AF647Stack_8int, 1)
    AF546Stack_8int_gaus = ndi.filters.gaussian_filter(AF546Stack_8int, 1)

    ###### Rank Mean bilateral filter
    i = 5
    struct = (np.mgrid[:i,:i][0] - np.floor(i/2))**2 + (np.mgrid[:i,:i][1] - np.floor(i/2))**2 <= np.floor(i/2)**2
    
    AF647Stack_8int_gaus_rank = np.empty_like(AF647Stack_8int_gaus)
    AF546Stack_8int_gaus_rank = np.empty_like(AF546Stack_8int_gaus)
    for pln in range(AF647Stack_8int_gaus.shape[0]):
        AF647Stack_8int_gaus_rank[pln] = filters.rank.mean_bilateral(AF647Stack_8int_gaus[pln], struct)
        AF546Stack_8int_gaus_rank[pln] = filters.rank.mean_bilateral(AF546Stack_8int_gaus[pln], struct)

    ###### Triangle Threshold
    AF647Stack_8int_gaus_rank_Triangle_thresh = filters.threshold_triangle(AF647Stack_8int_gaus_rank)
    AF647Stack_8int_gaus_rank_Triangle = AF647Stack_8int_gaus_rank > AF647Stack_8int_gaus_rank_Triangle_thresh

    AF546Stack_8int_gaus_rank_Triangle_thresh = filters.threshold_triangle(AF546Stack_8int_gaus_rank)
    AF546Stack_8int_gaus_rank_Triangle = AF546Stack_8int_gaus_rank > AF546Stack_8int_gaus_rank_Triangle_thresh

    ###### Watershed
    AF546CenterPointArrays = np.empty_like(AF546Stack_8int_gaus_rank_Triangle)
    AF647CenterPointArrays = np.empty_like(AF647Stack_8int_gaus_rank_Triangle)

    for pln in range(AF546Stack_8int_gaus_rank_Triangle.shape[0]):
        AF647Stack_8int_gaus_rank_Triangle_image = AF647Stack_8int_gaus_rank_Triangle[pln]
        AF546Stack_8int_gaus_rank_Triangle_image = AF546Stack_8int_gaus_rank_Triangle[pln]
        
        AF647distancetransform = ndi.distance_transform_edt(AF647Stack_8int_gaus_rank_Triangle_image)
        AF546distancetransform = ndi.distance_transform_edt(AF546Stack_8int_gaus_rank_Triangle_image)

        AF647Local_max = feature.peak_local_max(AF647distancetransform, indices = False)
        AF546Local_max = feature.peak_local_max(AF546distancetransform, indices = False)

        AF647CenterPointArrays[pln] = AF647Local_max
        AF546CenterPointArrays[pln] = AF546Local_max
    
    AF647CenterPointArrays_items, num = ndi.label(AF647CenterPointArrays)
    AF546CenterPointArrays_items, num = ndi.label(AF546CenterPointArrays)

    Conn26_ndarray = ndi.generate_binary_structure(3,3) # 26 connectivity.
    
    # AF647watershed_image = morphology.watershed(AF647Stack_8int_gaus_MIP_rank_Triangle, AF647Local_array_mask)
    # AF546watershed_image = morphology.watershed(AF546Stack_8int_gaus_MIP_rank_Triangle, AF546Local_array_mask)

    AF647watershed_image = morphology.watershed(AF647Stack_8int_gaus_rank_Triangle, AF647CenterPointArrays_items, connectivity = Conn26_ndarray)
    AF546watershed_image = morphology.watershed(AF546Stack_8int_gaus_rank_Triangle, AF546CenterPointArrays_items, connectivity = Conn26_ndarray)

    print("Bilateral mean + Triangle Thresholded Images complete for ", CziName)
    print("Watershed Found ", len(np.unique(AF647watershed_image)), " spots in the 647 channel.")
    print("Watershed Found ", len(np.unique(AF546watershed_image)), " spots in the 546 channel.")

    #### Creating the Nuclei edges
    CellEdge = np.zeros_like(combined_405_488_watershed_image_noborders) #make an array the same size and data type as the watershed but filled with zeros
    for cell in np.unique(combined_405_488_watershed_image_noborders):
        cell_mask = combined_405_488_watershed_image_noborders==cell
        ErodedCellMask = ndi.binary_erosion(cell_mask, iterations=1) # Increase iterations to make boundary wider!
        edge_mask = np.logical_xor(cell_mask, ErodedCellMask) #  Create the cell edge mask
        CellEdge[edge_mask] = cell
    CellEdge_mask = np.ma.array(CellEdge, mask=CellEdge==0)

    fig = plt.figure()
    fig.suptitle('Stack Bilateral mean + Thresholding Output' + str(CziName))
    ax = plt.subplots(figsize=(20, 10))
    plt.subplot(2,4,1)
    plt.title('AF647Stack_8int_gaus')
    plt.imshow(np.amax(AF647Stack_8int_gaus, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,2)
    plt.title('AF647Stack_8int_gaus_rank')
    plt.imshow(np.amax(AF647Stack_8int_gaus_rank, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,3)
    plt.title('AF647Stack_8int_gaus_rank_Triangle')
    plt.imshow(np.amax(AF647Stack_8int_gaus_rank_Triangle, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,4)
    plt.title('AF647watershed_image')
    plt.imshow(np.amax(AF647watershed_image, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,5)
    plt.title('AF546Stack_8int_gaus')
    plt.imshow(np.amax(AF546Stack_8int_gaus, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,6)
    plt.title('AF546Stack_8int_gaus_rank')
    plt.imshow(np.amax(AF546Stack_8int_gaus_rank, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,7)
    plt.title('AF546Stack_8int_gaus_rank_Triangle')
    plt.imshow(np.amax(AF546Stack_8int_gaus_rank_Triangle, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,4,8)
    plt.title('AF546watershed_image')
    plt.imshow(np.amax(AF546watershed_image, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.savefig('DNASpot_Filter_Thresholding' + str(CziName) + '.png')
    plt.close()

    return(AF647watershed_image, AF546watershed_image, AF647Stack_8int_gaus_rank_Triangle, AF546Stack_8int_gaus_rank_Triangle)

def MIP_Threshold_DNASpots_triangle_BilateralRankMean_647_546_488(CziName, AF647Stack, AF546Stack, AF488Stack, watershed_image_noborders):
    ###### Turn into 8 bit
    AF647Stack_8bit = img_as_ubyte(AF647Stack)
    AF546Stack_8bit = img_as_ubyte(AF546Stack)
    AF488Stack_8bit = img_as_ubyte(AF488Stack)

    ###### Small Gaussian Filter
    AF647Stack_8bit_gaus = ndi.filters.gaussian_filter(AF647Stack_8bit, 1)
    AF546Stack_8bit_gaus = ndi.filters.gaussian_filter(AF546Stack_8bit, 1)
    AF488Stack_8bit_gaus = ndi.filters.gaussian_filter(AF488Stack_8bit, 1)


    ###### Rank Mean bilateral filter
    
        ###### >>>> Make Empty files
    AF647Stack_8bit_gaus_rank = np.empty_like(AF647Stack_8bit_gaus)
    AF488Stack_8bit_gaus_rank = np.empty_like(AF488Stack_8bit_gaus)
    AF546Stack_8bit_gaus_rank = np.empty_like(AF546Stack_8bit_gaus)
        
        ###### >>>> Bilateral filter on Z stack
    i = 5
    struct = (np.mgrid[:i,:i][0] - np.floor(i/2))**2 + (np.mgrid[:i,:i][1] - np.floor(i/2))**2 <= np.floor(i/2)**2

    for pln in range(AF647Stack_8bit_gaus.shape[0]):
        AF647Stack_8bit_gaus_rank[pln] = filters.rank.mean_bilateral(AF647Stack_8bit_gaus[pln], struct)
        AF488Stack_8bit_gaus_rank[pln] = filters.rank.mean_bilateral(AF488Stack_8bit_gaus[pln], struct)
        AF546Stack_8bit_gaus_rank[pln] = filters.rank.mean_bilateral(AF546Stack_8bit_gaus[pln], struct)

    ###### MIP
    AF647Stack_8bit_gaus_rank_MIP = np.amax(AF647Stack_8bit_gaus_rank, axis = 0)
    AF546Stack_8bit_gaus_rank_MIP = np.amax(AF546Stack_8bit_gaus_rank, axis = 0)
    AF488Stack_8bit_gaus_rank_MIP = np.amax(AF488Stack_8bit_gaus_rank, axis = 0)

    ###### Save MIP Bilateral filter, triangle filter
    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize=(50, 50))
    fig = plt.figure()
    fig.suptitle('MIP Bilateral mean' + str(CziName))
    plt.subplot(1,3,1)
    plt.title('AF647Stack_8bit_gaus_rank_MIP')
    plt.imshow(AF647Stack_8bit_gaus_rank_MIP)
    plt.subplot(1,3,2)
    plt.title('AF546Stack_8bit_gaus_rank_MIP')
    plt.imshow(AF546Stack_8bit_gaus_rank_MIP)
    plt.subplot(1,3,3)
    plt.title('AF488Stack_8bit_gaus_rank_MIP')
    plt.imshow(AF488Stack_8bit_gaus_rank_MIP)
    plt.savefig('MIP_Bilateral_mean_' + str(CziName) + '.png')
    plt.close()


    ###### Triangle Threshold
    AF647Stack_8bit_gaus_MIP_rank_Triangle_thresh = filters.threshold_triangle(AF647Stack_8bit_gaus_rank_MIP)
    AF647Stack_8bit_gaus_MIP_rank_Triangle = AF647Stack_8bit_gaus_rank_MIP > AF647Stack_8bit_gaus_MIP_rank_Triangle_thresh

    AF546Stack_8bit_gaus_MIP_rank_Triangle_thresh = filters.threshold_triangle(AF546Stack_8bit_gaus_rank_MIP)
    AF546Stack_8bit_gaus_MIP_rank_Triangle = AF546Stack_8bit_gaus_rank_MIP > AF546Stack_8bit_gaus_MIP_rank_Triangle_thresh

    AF488Stack_8bit_gaus_MIP_rank_Triangle_thresh = filters.threshold_triangle(AF488Stack_8bit_gaus_rank_MIP)
    AF488Stack_8bit_gaus_MIP_rank_Triangle = AF488Stack_8bit_gaus_rank_MIP > AF488Stack_8bit_gaus_MIP_rank_Triangle_thresh

    ###### Watershed
    AF647distancetransform = ndi.distance_transform_edt(AF647Stack_8bit_gaus_MIP_rank_Triangle)
    AF546distancetransform = ndi.distance_transform_edt(AF546Stack_8bit_gaus_MIP_rank_Triangle)
    AF488distancetransform = ndi.distance_transform_edt(AF488Stack_8bit_gaus_MIP_rank_Triangle)

    AF647Local_max = feature.peak_local_max(AF647distancetransform, indices = False)
    AF546Local_max = feature.peak_local_max(AF546distancetransform, indices = False)
    AF488Local_max = feature.peak_local_max(AF488distancetransform, indices = False)

    AF647Local_max_mask = np.ma.array(AF647Local_max, mask=AF647Local_max==0) #makes a mask so that I can visualize on top of the original image
    AF546Local_max_mask = np.ma.array(AF546Local_max, mask=AF546Local_max==0)
    AF488Local_max_mask = np.ma.array(AF488Local_max, mask=AF488Local_max==0)

    AF647Local_max_mask_array, AF647Local_max_mask_array_num_features = ndi.label(AF647Local_max_mask)
    AF546Local_max_mask_array, AF546Local_max_mask_array_num_features = ndi.label(AF546Local_max_mask)
    AF488Local_max_mask_array, AF488Local_max_mask_array_num_features = ndi.label(AF488Local_max_mask)

    AF647Local_array_mask = np.ma.array(AF647Local_max_mask_array, mask=AF647Local_max_mask_array==0) #make a segmentation mask
    AF546Local_array_mask = np.ma.array(AF546Local_max_mask_array, mask=AF546Local_max_mask_array==0)
    AF488Local_array_mask = np.ma.array(AF488Local_max_mask_array, mask=AF488Local_max_mask_array==0)

    AF647watershed_image = morphology.watershed(AF647Stack_8bit_gaus_MIP_rank_Triangle, AF647Local_array_mask)
    AF546watershed_image = morphology.watershed(AF546Stack_8bit_gaus_MIP_rank_Triangle, AF546Local_array_mask)
    AF488watershed_image = morphology.watershed(AF488Stack_8bit_gaus_MIP_rank_Triangle, AF488Local_array_mask)

    print("MIP Bilateral mean + Triangle Thresholded Images complete for ", CziName)
    print("Found ", AF647Local_max_mask_array_num_features, " spots in the 647 channel.")
    print("Found ", AF546Local_max_mask_array_num_features, " spots in the 546 channel.")
    print("Found ", AF488Local_max_mask_array_num_features, " spots in the 488 channel.")
    print("Watershed Found ", len(np.unique(AF647watershed_image)), " spots in the 647 channel.")
    print("Watershed Found ", len(np.unique(AF546watershed_image)), " spots in the 546 channel.")
    print("Watershed Found ", len(np.unique(AF488watershed_image)), " spots in the 488 channel.")

    #### Creating the Nuclei edges
    CellEdge = np.zeros_like(watershed_image_noborders) #make an array the same size and data type as the watershed but filled with zeros
    for cell in np.unique(watershed_image_noborders):
        cell_mask = watershed_image_noborders==cell
        ErodedCellMask = ndi.binary_erosion(cell_mask, iterations=1) # Increase iterations to make boundary wider!
        edge_mask = np.logical_xor(cell_mask, ErodedCellMask) #  Create the cell edge mask
        CellEdge[edge_mask] = cell
    CellEdge_mask = np.ma.array(CellEdge, mask=CellEdge==0)

    fig = plt.figure()
    fig.suptitle('MIP Bilateral mean + Thresholding Output' + str(CziName))
    ax = plt.subplots(figsize=(20, 10))
    plt.subplot(3,3,1)
    plt.title('AF647Stack_8bit_gaus_rank_MIP')
    plt.imshow(AF647Stack_8bit_gaus_rank_MIP)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(3,3,2)
    plt.title('AF647Stack_8bit_gaus_MIP_rank_Triangle')
    plt.imshow(AF647Stack_8bit_gaus_MIP_rank_Triangle)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(3,3,3)
    plt.title('AF647watershed_image')
    plt.imshow(AF647watershed_image)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(3,3,4)
    plt.title('AF546Stack_8bit_gaus_rank_MIP')
    plt.imshow(AF546Stack_8bit_gaus_rank_MIP)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(3,3,5)
    plt.title('AF546Stack_8bit_gaus_MIP_rank_Triangle')
    plt.imshow(AF546Stack_8bit_gaus_MIP_rank_Triangle)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(3,3,6)
    plt.title('AF546watershed_image')
    plt.imshow(AF546watershed_image)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(3,3,7)
    plt.title('AF488Stack_8bit_gaus_rank_MIP')
    plt.imshow(AF488Stack_8bit_gaus_rank_MIP)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(3,3,8)
    plt.title('AF488Stack_8bit_gaus_MIP_rank_Triangle')
    plt.imshow(AF488Stack_8bit_gaus_MIP_rank_Triangle)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(3,3,9)
    plt.title('AF488watershed_image')
    plt.imshow(AF488watershed_image)
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.savefig('DNASpot_Filter_Thresholding' + str(CziName) + '.png')
    plt.close()

    return(AF647watershed_image, AF546watershed_image, AF488watershed_image, AF647Stack_8bit_gaus_MIP_rank_Triangle, AF546Stack_8bit_gaus_MIP_rank_Triangle, AF488Stack_8bit_gaus_MIP_rank_Triangle, AF647Stack_8bit_gaus_rank_MIP, AF546Stack_8bit_gaus_rank_MIP, AF488Stack_8bit_gaus_rank_MIP)

def NucleiSegmentation_MIP_Dapi_Lamina(CziName, AF405Stack):
    
    AF405Stack_8bit = img_as_ubyte(AF405Stack)
    AF405Stack_8bit_MIP = np.amax(AF405Stack_8bit, axis = 0)

    #### PART 1: Find centers with a conservative threshold
    ### First perform the conservative threshold on every point in the Z axis with the Otsu threshold
    AF405Stack_8bit_GausFilt = ndi.filters.gaussian_filter(AF405Stack_8bit_MIP, 1)
    otsuthresh = filters.threshold_otsu(AF405Stack_8bit_GausFilt)
    AF405Stack_8bit_GausFilt_localtheshold = AF405Stack_8bit_GausFilt > otsuthresh

    #### Then Call centers 

    ### Distance transform
    distancetransform_combined_final = ndi.distance_transform_edt(AF405Stack_8bit_GausFilt_localtheshold)

    #### smoothen the distance transform
    distancetransform_combined_final_gaus = ndi.filters.gaussian_filter(distancetransform_combined_final, 10)

    #### Retrieve the local maxima from the distance transform
    Local_max = feature.peak_local_max(distancetransform_combined_final_gaus, indices = False, min_distance = 40)
    Local_max_bigger = ndi.filters.maximum_filter(Local_max, size=20)
    Local_max_mask = np.ma.array(Local_max_bigger, mask=Local_max_bigger==0) #makes a mask so that I can visualize on top of the original image

    #Add that mask back into the watershed image
    CenterPointArrays = Local_max_bigger

    #### PART 2: Now make a mask with a permissive threshold that goes all the way to the edges of the nuclei.
    image_GausFilt = ndi.filters.gaussian_filter(AF405Stack_8bit_MIP, 20)
    localthresh = filters.threshold_local(image_GausFilt, 41)
    image_GausFilt_localtheshold = image_GausFilt > localthresh
    image_GausFilt_localtheshold_dilate = morphology.binary_dilation(image_GausFilt_localtheshold, selem = disk(6))
    EdgeMask = image_GausFilt_localtheshold_dilate

    #### Part 3: watershed
    CenterPointArrays_items, num = ndi.label(CenterPointArrays)

    ## Watershed
    watershed_image = morphology.watershed(~AF405Stack_8bit_GausFilt, CenterPointArrays_items, mask = EdgeMask)

    #### Part 4: Clear borders
    Watershed_ClearBorders = segmentation.clear_border(watershed_image)

    ### Itemize
    labeled_array_segmentation, num_features_seg = ndi.label(Local_max_mask)

    ## Visualize
    fig, axs = plt.subplots(nrows = 7, ncols = 5, figsize=(10, 10))
    fig.suptitle('MIP Otsu mean + Thresholding Output' + str(CziName))
    plt.subplot(2,2,1)
    plt.title('Centerpoints')
    plt.imshow(AF405Stack_8bit_MIP)
    plt.imshow(CenterPointArrays,  cmap = 'Reds', alpha = 0.4)
    plt.subplot(2,2,2)
    plt.title('EdgeMask')
    plt.imshow(EdgeMask, alpha = 0.2, cmap = 'Greys')
    plt.subplot(2,2,3)
    plt.title('Watershed')
    plt.imshow(watershed_image)
    plt.subplot(2,2,4)
    plt.title('Watershed No borders')
    plt.imshow(Watershed_ClearBorders)
    plt.savefig(str(str(CziName) + '_Watershed.png'))
    plt.close()

    print("Printed PDF of segmentation results for ", CziName)
    print("segmented ", num_features_seg, "cells in ", CziName, ".")

    return(Watershed_ClearBorders)

def Better_Threshold_DNASpots_triangle(CziName, AF647Stack, AF546Stack, combined_405_488_watershed_image_noborders):

    ###### Small Gaussian Filter
    AF647Stack_gaus = ndi.filters.gaussian_filter(AF647Stack, 1)
    AF546Stack_gaus = ndi.filters.gaussian_filter(AF546Stack, 1)

    ###### Triangle Threshold
    AF647Stack_gaus_Triangle_thresh = filters.threshold_triangle(AF647Stack_gaus) + 2
    AF647Stack_gaus_Triangle = AF647Stack_gaus > AF647Stack_gaus_Triangle_thresh

    AF546Stack_gaus_Triangle_thresh = filters.threshold_triangle(AF546Stack_gaus) + 2
    AF546Stack_gaus_Triangle = AF546Stack_gaus > AF546Stack_gaus_Triangle_thresh

    ###### Watershed
    AF546CenterPointArrays = np.empty_like(AF647Stack_gaus_Triangle)
    AF647CenterPointArrays = np.empty_like(AF546Stack_gaus_Triangle)

    for pln in range(AF546Stack_gaus_Triangle.shape[0]):
        AF647Stack_gaus_Triangle_image = AF647Stack_gaus_Triangle[pln]
        AF546Stack_gaus_Triangle_image = AF546Stack_gaus_Triangle[pln]
        
        AF647distancetransform = ndi.distance_transform_edt(AF647Stack_gaus_Triangle_image)
        AF546distancetransform = ndi.distance_transform_edt(AF546Stack_gaus_Triangle_image)

        AF647Local_max = feature.peak_local_max(AF647distancetransform, indices = False)
        AF546Local_max = feature.peak_local_max(AF546distancetransform, indices = False)

        AF647CenterPointArrays[pln] = AF647Local_max
        AF546CenterPointArrays[pln] = AF546Local_max
    
    AF647CenterPointArrays_items, num = ndi.label(AF647CenterPointArrays)
    AF546CenterPointArrays_items, num = ndi.label(AF546CenterPointArrays)

    Conn26_ndarray = ndi.generate_binary_structure(3,3) # 26 connectivity.
    
    # AF647watershed_image = morphology.watershed(AF647Stack_8int_gaus_MIP_rank_Triangle, AF647Local_array_mask)
    # AF546watershed_image = morphology.watershed(AF546Stack_8int_gaus_MIP_rank_Triangle, AF546Local_array_mask)

    AF647watershed_image = morphology.watershed(AF647Stack_gaus_Triangle, AF647CenterPointArrays_items, connectivity = Conn26_ndarray)
    AF546watershed_image = morphology.watershed(AF546Stack_gaus_Triangle, AF546CenterPointArrays_items, connectivity = Conn26_ndarray)
    

    print("Bilateral mean + Triangle Thresholded Images complete for ", CziName)
    print("Watershed Found ", len(np.unique(AF647watershed_image)), " spots in the 647 channel.")
    print("Watershed Found ", len(np.unique(AF546watershed_image)), " spots in the 546 channel.")

    #### Creating the Nuclei edges
    CellEdge = np.zeros_like(combined_405_488_watershed_image_noborders) #make an array the same size and data type as the watershed but filled with zeros
    for cell in np.unique(combined_405_488_watershed_image_noborders):
        cell_mask = combined_405_488_watershed_image_noborders==cell
        ErodedCellMask = ndi.binary_erosion(cell_mask, iterations=1) # Increase iterations to make boundary wider!
        edge_mask = np.logical_xor(cell_mask, ErodedCellMask) #  Create the cell edge mask
        CellEdge[edge_mask] = cell
    CellEdge_mask = np.ma.array(CellEdge, mask=CellEdge==0)

    fig = plt.figure()
    fig.suptitle('MIP Bilateral mean + Thresholding Output' + str(CziName))
    ax = plt.subplots(figsize=(20, 10))
    plt.subplot(2,3,1)
    plt.title('AF647Stack_gaus')
    plt.imshow(np.amax(AF647Stack_gaus, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,3,2)
    plt.title('AF647Stack_gaus_Triangle')
    plt.imshow(np.amax(AF647Stack_gaus_Triangle, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,3,3)
    plt.title('AF647watershed_image')
    plt.imshow(np.amax(AF647watershed_image, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,3,4)
    plt.title('AF546Stack_gaus')
    plt.imshow(np.amax(AF546Stack_gaus, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,3,5)
    plt.title('AF546Stack_gaus_Triangle')
    plt.imshow(np.amax(AF546Stack_gaus_Triangle, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.subplot(2,3,6)
    plt.title('AF546watershed_image')
    plt.imshow(np.amax(AF546watershed_image, axis=0))
    plt.imshow(CellEdge_mask, cmap = 'YlOrRd')
    plt.savefig('DNASpot_Filter_Thresholding' + str(CziName) + '.png')
    plt.close()

    return(AF647watershed_image, AF546watershed_image, AF647Stack_gaus, AF647Stack_gaus_Triangle, AF546Stack_gaus, AF546Stack_gaus_Triangle)


def GausMipForWeka2Dclassification_Input(ProjectData, NotebookResultsPath):
    
    os.chdir(NotebookResultsPath)

    if path.exists('MIPS') == False : 
        os.mkdir('MIPS')
    os.chdir('MIPS')
    if path.exists('AF647') == False : 
        os.mkdir('AF647')
    if path.exists('AF546') == False :
        os.mkdir('AF546')
    if path.exists('AF488') == False :
        os.mkdir('AF488')

    assert os.path.exists(ProjectData), "The project data directory " + str(ProjectData) + 'does not exist!' #if can't find the directory produces this error messaage
    
    ###### Import the filelist
    filelist = os.listdir(ProjectData) #list of file names

    ###### Add any file names that have .czi to a list target files
    target_files = []
    for fname in filelist:
        if fname.endswith('.czi'):
            target_files.append(fname)
        else:
             print("This file is not a CZI file - get outta here :", fname)
    ###### Loop through czi files 
    for czi in target_files:
        
        CziName = czi

        ###### split channels
        AF405Stack, AF488Stack, AF647Stack, AF546Stack = CZIMetadatatoDictionaries(ProjectData, CziName)

        ###### Only MIPS make for stacks that are not empty

        if str(AF647Stack) != 'empty' :
        
            ###### gaussian
            AF647Stack_gaus = ndi.filters.gaussian_filter(AF647Stack, 1)
            ###### MIP
            AF647Stack_gaus_MIP = np.amax(AF647Stack_gaus, axis = 0)
            
            ###### Save output as the czi name
            AF647_Path = str(NotebookResultsPath) + '/MIPS/AF647/'
            NameString = str(AF647_Path + CziName + '__gaus_MIP_647.png')
            plt.imsave(NameString, AF647Stack_gaus_MIP)
        else : 
            print('AF647 is empty')
        
        if str(AF488Stack) != 'empty' :
            
            AF488Stack_gaus = ndi.filters.gaussian_filter(AF488Stack, 1)
            
            AF488Stack_gaus_MIP = np.amax(AF488Stack_gaus, axis = 0)
            
            AF488_Path = str(NotebookResultsPath) + '/MIPS/AF488/'
            NameString = str(AF488_Path + CziName + '__gaus_MIP_488.png')
            plt.imsave(NameString, AF488Stack_gaus_MIP)
        
        else : 
            print('AF488 is empty')

        if str(AF546Stack) != 'empty' :

            AF546Stack_gaus = ndi.filters.gaussian_filter(AF546Stack, 1)

            AF546Stack_gaus_MIP = np.amax(AF546Stack_gaus, axis = 0)

            AF546_Path = str(NotebookResultsPath) + '/MIPS/AF546/'
            NameString = str(AF546_Path + CziName + '__gaus_MIP_546.png')
            plt.imsave(NameString, AF546Stack_gaus_MIP)
        else : 
            print('AF546 is empty')

    return()

def GausMipForWeka3Dclassification_Input(ProjectData, NotebookResultsPath):
    
    os.chdir(NotebookResultsPath)

    if path.exists('MIPS') == False : 
        os.mkdir('MIPS')
    os.chdir('MIPS')
    if path.exists('AF647') == False : 
        os.mkdir('AF647')
    if path.exists('AF546') == False :
        os.mkdir('AF546')
    if path.exists('AF488') == False :
        os.mkdir('AF488')

    assert os.path.exists(ProjectData), "The project data directory " + str(ProjectData) + 'does not exist!' #if can't find the directory produces this error messaage
    
    ###### Import the filelist
    filelist = os.listdir(ProjectData) #list of file names

    ###### Add any file names that have .czi to a list target files
    target_files = []
    for fname in filelist:
        if fname.endswith('.czi'):
            target_files.append(fname)
        else:
             print("This file is not a CZI file - get outta here :", fname)
    ##### Loop through czi files 
    for czi in target_files:
        
        CziName = czi
        
        print(str(CziName))

        ##### split channels
        AF405Stack, AF488Stack, AF647Stack, AF546Stack = CZIMetadatatoDictionaries(ProjectData, CziName)

        ##### Only MIPS make for stacks that are not empty

        if str(AF647Stack) != 'empty' :
        
            ###### gaussian
            AF647Stack_gaus = ndi.filters.gaussian_filter(AF647Stack, 1)
            
            ###### Save output as the czi name
            AF647_Path = str(NotebookResultsPath) + '/stacks/AF647/'
            NameString = str(AF647_Path + CziName + '__gaus_647.tif')
            imwrite(NameString, AF647Stack_gaus, photometric='minisblack')
            #imsave(NameString, AF647Stack_gaus)
        else : 
            print('AF647 is empty')
        
        if str(AF488Stack) != 'empty' :
            
            AF488Stack_gaus = ndi.filters.gaussian_filter(AF488Stack, 1)
            
            
            AF488_Path = str(NotebookResultsPath) + '/stacks/AF488/'
            NameString = str(AF488_Path + CziName + '__gaus_488.tif')
            imwrite(NameString, AF488Stack_gaus, photometric='minisblack')
            #plt.imsave(NameString, AF488Stack_gaus)
        
        else : 
            print('AF488 is empty')

        if str(AF546Stack) != 'empty' :

            AF546Stack_gaus = ndi.filters.gaussian_filter(AF546Stack, 1)

            AF546_Path = str(NotebookResultsPath) + '/stacks/AF546/'
            NameString = str(AF546_Path + CziName + '__gaus_546.tif')
            imwrite(NameString, AF546Stack_gaus, photometric='minisblack')
            #plt.imsave(NameString, AF546Stack_gaus)
        else : 
            print('AF546 is empty')

    return()

    ###test

