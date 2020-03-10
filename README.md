# reading_czi_stack_videos_GH

**Purpose**: Building code to trace MS2 spots from 3D confocal videos. 

**Overall the steps are as follows**
1. Read in CZI files and associated metadata for analzying 
2. isolate and label MS2 spots through time 
3. isolate and label nuclei through time 
4. associate spots with nuclei 
5. analyze MS2 spot location in time and the "bursting frequency".

## Directory Guide

- `imageAnalysis_jenna`: This directory hold image analysis work from Jenna Haines, who was trying to simulate similar work
- `img/`: images from notebooks
-  `presentations`: presentations to help understand background
- `videos`: 
- `notebooks`:	see below

### Notebook Guide
- `00_exploring_color.ipynb`: Notebook exploring reading in czi file and color options
- `01_explore.ipynb`: How to create Z-stacks, movies, and basic thresholding (remove) 
- `01_explore_ST.ipynb`:	How to create Z-stacks, movies, and Samantha added more by exploring how to use thresholding to isolate **MS2 spots**
-  `2_explore_nuclei_segment.ipynb`: Using work from Jenna's code to isolate **nuclei**
-  `2_tracking_dots_ST.ipynb`: Isolating and **clustering MS2 spots**
-  `3_MS2_video_ST.ipynb`: Making videos of the work
-  `3_nuclei_vis_function.ipynb`:	Making videos of isolated **nuclei**
-  `4_time_component_clustering_ST.ipynb`: Looking into how the **MS2 spots** cluster according to time.


More Data: https://drive.google.com/drive/folders/1gz0ZDjHVI20MsKUmDASuKc8mWiT32_rU
