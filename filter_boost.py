import numpy as np
import h5py
import scipy.sparse
import scipy.io
from constants import *
import os
import pickle

# frame length, which also dictates the delay being frame capture and feedback
# because of forward_fit
# which isn't even in the report...
flen = DEE
flen_2 = 3
dt = EPSILON
st = 0.75  # kind of equivalent to sigma

res_dict = {}

ground_truth = scipy.io.loadmat('GroundTruth_Eynsham_40meters.mat')['ground_truth']

for fname in os.listdir("good"):
### Get matches from confusion matrix ###

    # load the confusion matrix
    dname = "dataset"
    print("opening file %s" % fname)
    h5f = h5py.File("good/"+fname, 'r')
    conf_matrix = h5f[dname][:]
    h5f.close()
    print("procesing layer")

    # grab the testing matrix from the confusion matrix
    test_matrix = conf_matrix[0:4789, 4789:9575]
    # the min score is the best match
    b = np.argmin(test_matrix, axis=0)

    # Percentage of top matches used in the vibration calculation, allows the occasional outlier
    inlier_fraction = 5/6.0

    matches = np.zeros(int(b.size - flen + flen_2))
    stable_count = 0

    # WHY NOT FILTER AROUND? Change to get same results but neater?
    for i in range(0, b.size - flen):

        match_index = int(i + flen_2)

        # Check that the match being considered is continous with those around it
        vibrations = np.abs( np.diff(b[i:i + flen]) )
        sorted_vib = np.sort(vibrations)
        max_diff = np.max(sorted_vib[ 0 : int(np.round(inlier_fraction * flen)) ])
        stable = max_diff <= dt

        # linear regression to get slope of fit
        pt = np.polyfit( np.arange(0, flen), b[i:i + flen], 1)
        # This is the slope, because highest powers first
        velocity = pt[0]

        # forward match with a tolerance of -1 and +1
        # absolute value to check going forwards or backwards
        forward_match = np.abs(velocity - 1) < st or np.abs(velocity + 1) < st

        if stable and forward_match:
            # smooth the value based off of those around it
            matches[match_index] = pt[1] + pt[0] * 0.5 * flen

            for j in range(1, flen_2 + 1):
                back_chk = match_index - j
                front_chk = match_index + j
                # fill in the zero (default) values if possible
                if matches[back_chk] == 0:
                    matches[back_chk] = b[back_chk]
                # fill in base values for future vals
                if front_chk < 4783:
                    matches[front_chk] = b[front_chk]


    ### Compare to ground truth ###
    print("zeros")
    print(np.where(matches == 0)[0].size)
    print("comparing to ground truth")
    start_first = 1
    end_first = 4788
    len_first = end_first - start_first + 1
    start_second = 4789
    end_second = 9574
    len_second = end_second - start_second + 1
    half_matrix = 4785

    ground_matrix = np.zeros((len_second, len_first))

    tp_num = 0
    tp_value = []
    fp_num = 0
    fp_value = []

    for ground_idx in range(start_second, end_second):
        value_ground = ground_truth[ground_idx, :]
        value_fit = value_ground.toarray().flatten().nonzero()[0]
        # only store those in first round
        value_fit2 = value_fit[ np.where(value_fit < end_first)[0].astype(int) ]
        value_fit3 = value_fit2 - start_first + 1
        value_fit4 = value_fit3[ np.where(value_fit3 > 0)[0].astype(int) ]

        matrix_idx = ground_idx - start_second + 1
        ground_matrix[matrix_idx, value_fit4] = 1

    for truth_idx in range(0, matches.size):

        ground_row = ground_truth[truth_idx+end_first, :]
        ground_row_idx = ground_row.toarray().flatten().nonzero()[0]

        if matches[truth_idx] != 0:
            truth_va = np.round(matches[truth_idx])
            if np.any(ground_row_idx == np.round(truth_va)):
                tp_num = tp_num + 1
                tp_value = [tp_value, truth_idx]
            else:
                fp_num = fp_num + 1
                fp_value = [fp_value, truth_idx]

    precision = tp_num / float(tp_num + fp_num)
    print(precision)
    recall = tp_num / float(b.size)
    print(recall)
    res_dict[fname] = (precision, recall)

pickle.dump(res_dict, open("filter_res.p", "wb"))
