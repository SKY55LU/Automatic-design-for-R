"""if possible,
please run the code under the following environment:
python 3.6, pandas 0.20.3, numpy 1.19.5, geomdl 5.3.1, matplotlib 3.3.4

with support of this code, you can generate the NURBS curve with five control points
and the 50-bit pixel matrix of the curve

this code is used to generate the 10000 nurbs CURVE
two important examples are shown in th end"""

import math
import pandas as pd
import numpy as np
from geomdl import BSpline
from geomdl import utilities
from numpy import random
import matplotlib.pyplot as plt


def pixel_generate(variable):

    """ Generate the pixel matrix of the NURBS curve"""

    control_points = [[0, 0] for i in range(math.ceil(len(variable) / 2))]
    for i in range(math.ceil(len(variable) / 2)):
        control_points[i][0] = variable[i]
        control_points[i][1] = variable[i + math.ceil(len(variable) / 2)]

    # Generate curve
    curve = BSpline.Curve()
    curve.degree = 3
    curve.ctrlpts = control_points
    # generate_knot_vector
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))

    # set samples
    curve.delta = 0.0001

    # evaluate curve
    curve.evaluate()

    # Sampling points are screened and duplicate blocks are removed
    points = pd.DataFrame(curve.evalpts)
    points_fit = points[(points[0] >= 0) & (points[0] < 10) & (points[1] >= 0) & (points[1] < 5)]
    # print(points_fit)
    ceil = np.floor(points_fit.values)
    # print(ceil)
    ceil_df = pd.DataFrame(ceil)
    ceil_df.drop_duplicates(keep='first', inplace=True)

    # map to pixel
    pixel = np.zeros((10, 5))
    for j in range(ceil_df.shape[0]):
        a = int(ceil_df.iloc[j, 0])
        b = int(ceil_df.iloc[j, 1])
        pixel[a, b] = 1

    #  stored as a one-dimensional code
    pixel_list = sum(pixel.reshape(1, 50).tolist(), [])

    # if possible, we can see the curve and pixel
    points_1 = points - 0.5
    points_1.plot(x=1, y=0, xlim=[-0.5, 5.5], ylim=[10.5, -0.5])
    cmap = plt.cm.inferno # set color map
    img = plt.imshow(pixel, cmap=cmap)
    plt.colorbar()
    plt.show()
    # print(pixel_list)

    return pixel_list


##################################################################

def nurbs_pixel(size):

    """ Define the region of control points and save the results"""

    control_points = random.random(size=(size, 10))
    control_points_after = random.random(size=(size, 10))
    pixel_all = list()
    control_points_all = list()

    for j in range(0, 9):
        for i in range(size):
            control_points_after[i][0:5] = control_points[i][0:5] * 11
            control_points_after[i][5:10] = control_points[i][5:10] * 6
            control_points_after[i][0] = 5
            control_points_after[i][5] = 0
            # print(control_points[i])
            pixel = pixel_generate(control_points_after[i])
            # print(control_points_after[i])
            control_points_all.append(control_points_after[i])
            pixel_all.append(pixel)

    pixel_df = pd.DataFrame(pixel_all)
    pixel_df.drop_duplicates(keep='first', inplace=True)

    control_points_df = pd.DataFrame(control_points_all)
    control_points_df.drop_duplicates(keep='first', inplace=True)

    pixel_df.to_csv('./NURBS_CURVE.csv')
    control_points_df.to_csv('./NURBS_CONTROL_POINTS.csv')

# # define the number of NURBS curve
# nurbs_pixel(10000)

# # three examples
pixel_generate([5, 9.3, 1.8, 10.2, 9.1, 0, 0.3, 4.4, 2.5, 1.1])
pixel_generate([5, 11, 5.56, 0, 1.209, 0, 2.88, 6, 1.87, 0])
