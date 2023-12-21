import open3d as o3d
import numpy as np
import math as m
import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd
import os
import stat
import sys
import csv
import json
from PIL import Image
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter1d
np.set_printoptions(threshold=sys.maxsize)

def curvaturePoints(b, c, d):
  temp = c[0]**2 + c[1]**2
  bc = (b[0]**2 + b[1]**2 - temp) / 2
  cd = (temp - d[0]**2 - d[1]**2) / 2
  det = (b[0] - c[0]) * (c[1] - d[1]) - (c[0] - d[0]) * (b[1] - c[1])

  if abs(det) < 1.0e-10:
    return 0.0

  # Center of circle
  cx = (bc*(c[1] - d[1]) - cd*(b[1] - c[1])) / det
  cy = ((b[0] - c[0]) * cd - (c[0] - d[0]) * bc) / det

  radius = ((cx - b[0])**2 + (cy - b[1])**2)**.5

  return 1.0/radius
def isAtBorder(x, y, impts):
    if impts[x, y] == 0:
        return False
    else:
        (sx, sy) = np.shape(impts)
        rx0 = -1
        rx1 = 1
        ry0 = -1
        ry1 = 1
        if x == 0:
            rx0 = 0
        if x == sx-1:
            rx1 = 0
        if y == 0:
            ry0 = 0
        if y == sy-1:
            ry1 = 0
        sum_s = 0
        for idx in range(rx0, rx1+1):
            for idy in range(ry0, ry1 + 1):
                if impts[x + idx, y + idy] == 0:
                    sum_s += 1
        return sum_s not in [0, 1, 5]
def crownOfPoints(border, sorted):
    s = 1
    lastPoint = sorted[-1]
    crownPoints = []
    while len(crownPoints) == 0:
        for idx in range(-s, s + 1):
            for idy in range(-s, s + 1):
                if (idx, idy) != (0, 0):
                    new_point = [lastPoint[0] + idx, lastPoint[1] + idy]
                    if (new_point in border) and (new_point not in sorted):
                        crownPoints.append(new_point)
        s += 1
    idx = 0
    max_point = crownPoints[0]
    for id in range(1, len(crownPoints)):
        if crownPoints[id][1] > max_point[1]:
            max_point = crownPoints[id]
            idx = id
    sorted.append(crownPoints[idx])
def sortBorder(border, dp, sx):
    l = len(border)
    sx2 = int(sx/2)
    d = 1
    # Choose first point
    idx = -1
    for i in range(l):
        if abs(border[i][d] - sx2) <= int(dp/2):
            if idx == -1:
                idx = i
                max0 = border[i][d]
            else:
                if border[i][d] < max0:
                    idx = i
                    max0 = border[i][d]
    if idx == -1:
        print("Error! Initial border point not found")
    else:
        sorted = [border[idx]]

    # Sort border
    for i in range(1, l):
        crownOfPoints(border, sorted)
    return sorted
def sortBorder2(border, dp, ymin, ymax):
    l = len(border)
    ymin = int((ymin+ymax)/2)
    d = 1
    # Choose first point
    idx = -1
    for i in range(l):
        if abs(border[i][d] - ymin) <= int(dp/2):
            if idx == -1:
                idx = i
                max0 = border[i][d]
            else:
                if border[i][d] < max0:
                    idx = i
                    max0 = border[i][d]
    if idx == -1:
        print("Error! Initial border point not found")
    else:
        sorted = [border[idx]]

    # Sort border
    for i in range(1, l):
        crownOfPoints(border, sorted)
    return sorted
def shift_rows_by_highest_average(arr):
    # Find the row index with the highest average (ignoring NaN values)
    avg_values = np.nanmean(arr, axis=1)
    max_avg_row_index = np.nanargmax(avg_values)

    # Calculate the number of rows to shift
    num_rows = arr.shape[0]
    shift_amount = num_rows // 2 - max_avg_row_index

    # Roll the array to move the row with the highest average to the center
    shifted_array = np.roll(arr, shift_amount, axis=0)

    return shifted_array

def shift_rows_by_highest_average2(arr):
    # Find the row index with the highest average (ignoring NaN values)
    avg_values = np.nanmean(arr, axis=1)
    max_avg_row_index = np.nanargmax(avg_values)

    # Extract the row with the highest average
    central_row = arr[max_avg_row_index, :]

    # Identify the non-NaN values in each column
    non_nan_values = np.where(~np.isnan(arr), arr, np.inf)

    # Sort each column based on non-NaN values
    sorted_arr = np.sort(non_nan_values, axis=0)

    # Find the indices of the sorted central row values in each column
    central_row_indices = np.searchsorted(sorted_arr[:, max_avg_row_index], central_row)

    # Calculate the shift amount for each column
    shift_amounts = central_row_indices - max_avg_row_index

    # Roll the array to move the row with the highest average to the center for each column
    shifted_array = np.zeros_like(arr)
    for col in range(arr.shape[1]):
        shifted_array[:, col] = np.roll(arr[:, col], shift_amounts[col])

    return shifted_array
def calculate_circle_length(positions):
    # Ensure the input array is a numpy array
    positions = np.asarray(positions)

    # Calculate the differences between consecutive positions
    differences = np.diff(positions, axis=0)

    # Include the distance between the last and first position (closed circle)
    circular_differences = np.vstack([differences, positions[0] - positions[-1]])

    # Compute the sum of the Euclidean distances between consecutive positions
    circle_length = np.sum(np.linalg.norm(circular_differences, axis=1))

    return circle_length
def read_single_value_from_csv(file_path):
    """
    Read a single value from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - value: The single value in the CSV file.
    """
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            value = next(reader)[0]

            return value
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except Exception as e:
        print(f"Error: {e}")


# DIRECTORY AND FOLDERS
directory = "C:/Users/jorfo/OneDrive/Documents/EMBL/Nick_Project/" # Change to your computer path
folders = ["pescoid_100um", "pescoid_200um", "pescoid_300um", "pescoid_ctrl"] # Edit to add or remove folders to study

# PARAMETERS TO EDIT
dp = 30 # Number of taken points to compute curvature
lTime = 39 # Number of time frames
iniTime = 7 # Initial time (in hours)
dt = 0.5 # Time interval difference (in hours)
aspect_ratio = 0.3467 # microns per pixel
dw = 5 # Number of rows for the final plot

# CODE (USUALLY NOT MODIFIED)
border_length = -1
lCommon = 10000 # Common number of points for relative border length (You should not modify this)

for folder in folders:
    path = directory + folder + "/Morgana/"

    # Gather all the confinement folders. The code only lists the folders begining with "Confinement".
    confinement_folders = [fol for fol in os.listdir(path) if os.path.isdir(os.path.join(path, fol)) and fol.startswith("Confinement")]
    dh = m.ceil(len(confinement_folders)/dw)
    w = 0
    h = 0

    # Compute the maximum border length over all confinement folders
    # If we have already computed the maximum length, we save it as a csv file to avoid repeating calculations
    if os.path.exists(path + "max_length.csv"):
        max_length = int(read_single_value_from_csv(path + "max_length.csv"))
        print(f"The value in the CSV file is: {max_length}")
    else:
        max_length = 0
        for con_folder in confinement_folders:
            for t in range(0, lTime):
                print(f"Reading max lent at {folder} ({con_folder}): {t+1}/{lTime}")
                # Load mask
                pathT = path + con_folder + "/result_segmentation/"

                img = Image.open(pathT + con_folder + "_time-" + str(t + 1).zfill(3) + "_finalMask.tif")
                img.load()

                # Save mask values in np
                sx = img.size[0]
                sy = img.size[1]
                impts = np.zeros((sx, sy))

                ymax = -np.Inf
                ymin = np.Inf

                for x in range(sx):
                    for y in range(sy):
                        impts[x, y] = img.getpixel((x, y))
                        if impts[x, y] == 1:
                            ymin = min(ymin, y)
                            ymax = max(ymax, y)

                # Comput the border of the mask and sort it
                border_length = 0
                for x in range(sx):
                    for y in range(sy):
                        if isAtBorder(x, y, impts):
                            border_length += 1
                #max_length = max(max_length, int(border_length / aspect_ratio) + 1)
                max_length = max(max_length, border_length)
            print(max_length)
        with open(path + "max_length.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([max_length])

    # Define the figures to be printed
    fig_abs, ax_abs = plt.subplots(dw, dh, figsize=(40, 20))
    fig_abs.subplots_adjust(wspace=0.3, hspace=0.4)

    fig_rel, ax_rel = plt.subplots(dw, dh, figsize=(40, 20))
    fig_rel.subplots_adjust(wspace=0.3, hspace=0.4)

    # Loop over the confinement folders
    for con_folder in confinement_folders:
        if not os.path.exists(path + con_folder + "/plots/gif/"):
            os.makedirs(path + con_folder + "/plots/gif/")
        Kymograph_abs = []
        Kymograph_rel = np.zeros((lCommon, lTime))
        for t in range(0, lTime):
            print(folder + " (" + con_folder + "): " + str(t+1) + "/" + str(lTime))
            # Load mask
            pathT = path + con_folder + "/result_segmentation/"

            img = Image.open(pathT + con_folder + "_time-" + str(t+1).zfill(3) + "_finalMask.tif")
            img.load()

            # Save mask values in np
            sx = img.size[0]
            sy = img.size[1]
            impts = np.zeros((sx, sy))

            ymax = -np.Inf
            ymin = np.Inf

            for x in range(sx):
                for y in range(sy):
                    impts[x, y] = img.getpixel((x, y))
                    if impts[x, y] == 1:
                        ymin = min(ymin, y)
                        ymax = max(ymax, y)

            # Compute the border of the mask and sort it
            border = []
            for x in range(sx):
                for y in range(sy):
                    if isAtBorder(x, y, impts):
                        border.append([x, y])
            npBorder = np.array(border)

            sBorder = sortBorder2(border, dp, ymin, ymax)
            npsBorder = np.array(sBorder)
            lBorder = len(npBorder)
            border_length = max(border_length, calculate_circle_length(npsBorder) * aspect_ratio)

            # Compute curvature
            curvature = np.zeros(lBorder)
            for p in range(lBorder):
                b = npsBorder[p - dp, :]
                c = npsBorder[p, :]
                d = npsBorder[(p + dp) % lBorder, :]
                mid = (b+d)/2.0
                if impts[round(mid[0]), round(mid[1])] == 1:
                    sign = 1.0
                else:
                    sign = -1.0

                curvature[p] = sign * curvaturePoints(b, c, d)

            # Smooth curvature and interpolate
            maxg = max(abs(max(curvature)), abs(min(curvature)))
            fac = 1.0
            curvatureGI = np.interp(np.linspace(0, 1, lCommon), np.linspace(0, 1, lBorder), curvature)

            Kymograph_abs.append(curvature.copy())
            Kymograph_rel[:, t] = curvatureGI.copy()

        # Rearrange the relative kymograph so the highest average of curvature (including sign) is at the center of y axis
        Kymograph_rel = shift_rows_by_highest_average(Kymograph_rel)
        kymograph_array = np.full((max_length, lTime), np.nan)
        for i in range(lTime):
            lrow = len(Kymograph_abs[i])
            start = int((max_length - lrow)/2)
            kymograph_array[start:start+lrow, i] = np.interp(np.linspace(0, 1, lrow),
                                                             np.linspace(0, 1, lCommon),
                                                             Kymograph_rel[:, i])
        # Plot relative kymograph
        max_abs_value = np.max(np.abs(Kymograph_rel))

        im_rel = ax_rel[w, h].imshow(Kymograph_rel, aspect='auto', vmin=-max_abs_value, vmax=max_abs_value, cmap='bwr',
                                     interpolation='none', extent=[iniTime, iniTime + dt * (lTime-1), 0, 1])
        ax_rel[w, h].set_xlabel('Time (hpf)')
        ax_rel[w, h].set_ylabel('Relative border length')
        ax_rel[w, h].set_title(con_folder)

        # Plot absolute kymograph
        ax_abs[w, h].imshow(np.isnan(kymograph_array), cmap='binary', aspect='auto', interpolation='none', alpha=0.5,
                            extent=[iniTime, iniTime + dt * (lTime-1), 0, max_length / aspect_ratio])

        im_abs = ax_abs[w, h].imshow(kymograph_array, cmap='bwr', aspect='auto', interpolation='none',
                                     vmin=-max_abs_value, vmax=max_abs_value,
                                     extent=[iniTime, iniTime + dt * (lTime-1), 0, max_length / aspect_ratio])

        ax_abs[w, h].set_xlabel('Time (hpf)')
        ax_abs[w, h].set_ylabel('Absolute border length')
        ax_abs[w, h].set_title(con_folder)

        # Load MOrgAna data
        with open(path + con_folder + "/result_segmentation/" + con_folder + '_fluo_intensity.json', 'r') as fcc_file:
            data = json.load(fcc_file)
            ch0 = np.zeros(lTime)
            ch1 = np.zeros(lTime)
            ax_abs2 = ax_abs[w, h].twinx()
            ax_rel2 = ax_rel[w, h].twinx()
            ax_abs2.set_ylabel('ch0 Average')
            ax_rel2.set_ylabel('ch0 Average')
            for t in range(lTime):
                ch0[t] = data[t]["ch0_Average"]
                ch1[t] = data[t]["ch1_Average"]
            ax_abs2.plot(np.linspace(iniTime, iniTime + dt * (lTime-1), lTime), ch0, color='green')
            ax_rel2.plot(np.linspace(iniTime, iniTime + dt * (lTime - 1), lTime), ch0, color='green')

        w += 1
        if w == dw:
            w = 0
            h += 1

    # SAVING KYMOGRAPH COMBINATION
    fig_abs.savefig(path + "all_kymograph_with_fluorescence_ch0_absolute.pdf")
    fig_rel.savefig(path + "all_kymograph_with_fluorescence_ch0_relative.pdf")
    # In case you want to save if as a .png, remove the comment of the lines below
    # fig_rel.savefig(path + "all_kymograph_absolute.png", dpi=300)
    # fig_rel.savefig(path + "all_kymograph_relative.png", dpi=300)

