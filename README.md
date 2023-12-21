# Pescoid-curvatures
Python pipeline to understand pescoid border curvature

In this repository we have three codes that work to plot kymographs containg information  about the gastruloid border along time.

Code code1.py plots the kymographs of curvature
Code code2.py plots the kymographs of curvature, as well as a superposed graph of both ch0 and ch1 average expressions
Code code3.py plots the kymographs of local ch1 fluorescence (not 

The parameters to modify in the code are:

directory (string): directory where the folders are stored (write the path with / and not \)
folders (string list): list of folders that you want kymographs from, with format ["name_of_folder_1", "name_of_folder_2", ...]
dp (int): distance of the two points taken along the border to compute the curvature (we need 3 points, the initial and two at both sides. These two poins are both at distance dp from the border)
lTime (int): number of time frames
iniTime (float): initial time (in hours)
dt (float): time difference between frames (in hours)
aspect_ratio (float): microns per pixel
dw (int): number of rows when printing the kymograph

IMPORTANT! All 3 pipelines generate a csv file that contains the maximum length across all borders over all folders. This is done in order to avoid repeating computations in case you run the code more than once. However, if you add or substract folders, delete the max_length.csv, since the code needs to generate a new one.
