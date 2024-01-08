# Pescoid-curvatures
Python pipeline to understand pescoid border curvature

In this repository we have three codes that work to plot kymographs containg information  about the gastruloid border along time.

(1) Kymograph_of_curvature_from_mask_all_absolute_and_relative.py plots the kymographs of curvature
(2) Kymograph_of_curvature_and_avgfluorescence_from_mask.py plots the kymographs of curvature, as well as a superposed graph of both ch0 and ch1 average expressions
(3) Kymograph_of_fluorescence.py plots the kymographs of local ch1 fluorescence

The parameters to modify in the code are:

directory (string): directory where the folders are stored (write the path with / and not \)
folders (string list): list of folders that you want kymographs from, with format ["name_of_folder_1", "name_of_folder_2", ...]
dp (int): distance of the two points taken along the border to compute the curvature (we need 3 points, the initial and two at both sides. These two poins are both at distance dp from the border)
lTime (int): number of time frames
iniTime (float): initial time (in hours)
dt (float): time difference between frames (in hours)
aspect_ratio (float): microns per pixel
dw (int): number of rows when printing the kymograph
[only for code (3)] df (int): size of radius to take the average of fluorescence

IMPORTANT! All 3 pipelines generate a csv file that contains the maximum length across all borders over all folders. This is done in order to avoid repeating computations in case you run the code more than once. However, if you add or substract folders, delete the max_length.csv, since the code needs to generate a new one.

IMPORTANT! Perform first the code (1) and then (2) and (3). Code (1) generates a file that (3) uses to rearrange the fluorescence

RECOMMENDATIONS: use the same value for dp and df (approximately a 10% of the average perimeter length)
