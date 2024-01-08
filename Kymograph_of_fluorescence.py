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
def shift_rows(shift_amount, arr):
    # Roll the array to move the row with the highest average to the center
    shifted_array = np.roll(arr, shift_amount, axis=0)
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
def create_numpy_of_pixel_values(img):
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
    return sx, sy, ymin, ymax, impts


# DIRECTORY AND FOLDERS
directory = "C:/Users/jorfo/OneDrive/Documents/EMBL/Nick_Project/" # Change to your computer path
folders = ["pescoid_100um", "pescoid_200um", "pescoid_300um", "pescoid_ctrl"] # Edit to add or remove folders to study

# PARAMETERS TO EDIT
dp = 30 # Number of taken points to compute fluorescence
lTime = 39 # Number of time frames
iniTime = 7 # Initial time (in hours)
dt = 0.5 # Time interval difference (in hours)
aspect_ratio = 0.3467 # microns per pixel
dw = 5 # Number of rows for the final plot
df = 30 # Number of the square of fluorescence of length 2*df+1

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
                sx, sy, impts = create_numpy_of_pixel_values(img)

                # Compute the border of the mask and sort it
                border_length = 0
                for x in range(sx):
                    for y in range(sy):
                        if isAtBorder(x, y, impts):
                            border_length += 1
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

        # Loop over frames (here called time)
        for t in range(0, lTime):
            print(folder + " (" + con_folder + "): " + str(t+1) + "/" + str(lTime))
            # Load mask
            pathT = path + con_folder + "/result_segmentation/"
            img = Image.open(pathT + con_folder + "_time-" + str(t+1).zfill(3) + "_finalMask.tif")
            img.load()

            img_flu = Image.open(path + con_folder + "/" + con_folder + "_time-" + str(t + 1).zfill(3) + ".tif")
            img_flu.load()

            # Save mask values in np
            sx, sy, ymin, ymax, impts = create_numpy_of_pixel_values(img)
            _, _, _, _, impts_flu = create_numpy_of_pixel_values(img_flu)

            # Comput the border of the mask and sort it
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

            # Compute fluorescence
            fluorescence = np.zeros(lBorder)
            for p in range(lBorder):
                c = npsBorder[p, :]
                n_ones = 0.0
                flu_sum = 0.0
                for i in range(c[0] - df, c[0] + df + 1):
                    if 0 <= i < sx:
                        for j in range(c[1] - df, c[1] + df + 1):
                            if 0 <= j < sy:
                                if impts[i, j] == 1 and (i - c[0])**2 + (j - c[1])**2 <= df*df:
                                    flu_sum += float(impts_flu[i, j])
                                    n_ones += 1.0
                fluorescence[p] = flu_sum/n_ones

            # Interpolate fluorescence to a common length for the relative plot
            fac = 1.0
            fluorescenceGI = np.interp(np.linspace(0, 1, lCommon), np.linspace(0, 1, lBorder), fluorescence)

            Kymograph_abs.append(fluorescence.copy())
            Kymograph_rel[:, t] = fluorescenceGI.copy()

        # Rearrange the relative kymograph so the highest average of fluorescence (including sign) is at the center of y axis
        shift_amount = int(read_single_value_from_csv(path + con_folder + "/shift_amount.csv"))
        Kymograph_rel = shift_rows(shift_amount, Kymograph_rel)
        minf = min([min(line) for line in Kymograph_abs])
        maxf = max([max(line) for line in Kymograph_abs])

        kymograph_array = np.full((max_length, lTime), np.nan)
        for i in range(lTime):
            lrow = len(Kymograph_abs[i])
            start = int((max_length - lrow)/2)
            kymograph_array[start:start+lrow, i] = np.interp(np.linspace(0, 1, lrow),
                                                             np.linspace(0, 1, lCommon),
                                                             Kymograph_rel[:, i])
        # Plot relative kymograph
        max_abs_value = np.max(np.abs(Kymograph_rel))

        im_rel = ax_rel[w, h].imshow(Kymograph_rel, aspect='auto', vmin=minf, vmax=maxf, cmap='winter',
                                     interpolation='none', extent=[iniTime, iniTime + dt * (lTime-1), 0, 1])
        ax_rel[w, h].set_xlabel('Time (hpf)')
        ax_rel[w, h].set_ylabel('Relative border length')
        ax_rel[w, h].set_title(con_folder)
        plt.colorbar(im_rel, ax=ax_rel[w, h])

        # Plot absolute kymograph
        ax_abs[w, h].imshow(np.isnan(kymograph_array), cmap='binary', aspect='auto', interpolation='none', alpha=0.5,
                            extent=[iniTime, iniTime + dt * (lTime-1), 0, max_length / aspect_ratio])

        im_abs = ax_abs[w, h].imshow(kymograph_array, cmap='winter', aspect='auto', interpolation='none',
                                     vmin=minf, vmax=maxf,
                                     extent=[iniTime, iniTime + dt * (lTime-1), 0, max_length / aspect_ratio])

        ax_abs[w, h].set_xlabel('Time (hpf)')
        ax_abs[w, h].set_ylabel('Absolute border length')
        ax_abs[w, h].set_title(con_folder)
        plt.colorbar(im_abs, ax=ax_abs[w, h])

        w += 1
        if w == dw:
            w = 0
            h += 1

    # SAVING KYMOGRAPH COMBINATION
    fig_abs.savefig(path + "all_kymograph_flu_absolute.pdf")
    fig_rel.savefig(path + "all_kymograph_flu_relative.pdf")
    # In case you want to save if as a .png, remove the comment of the lines below
    # fig_rel.savefig(path + "all_kymograph_absolute.png", dpi=300)
    # fig_rel.savefig(path + "all_kymograph_relative.png", dpi=300)