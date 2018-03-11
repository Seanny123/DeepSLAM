import h5py
import scipy.io
from constants import *

flen = DEE
flen_2 = 3
dt = EPSILON
st = 0.75  # kind of equivalent to sigma

### Get matches from confusion matrix ###

# load the confusion matrix
fname = "C:\\Users\\saubin\\Desktop\\DeepSLAM\\conf_mat_smush_full_overfeat_10.h5"
dname = "dataset"
print("opening file")
h5f = h5py.File(fname, 'r')
conf_matrix = h5f[dname][:]
h5f.close()
print("processing layer")

# grab the testing matrix from the confusion matrix
test_matrix = conf_matrix[0:4789, 4789:9575]
# the min score is the best match
b = np.argmin(test_matrix, axis=0)

# Percentage of top matches used in the vibration calculation, allows the occasional outlier
inlier_fraction = 5/6.0

p = np.zeros(b.size)
matches = np.zeros(int(b.size - flen + flen_2))
max_diff = 0

for i in range(0, b.size - flen):

    match_index = int(i + flen_2)
    vibrations = np.abs( np.diff(b[i:i + flen]) )
    sorted_vib = np.sort(vibrations)
    max_diff = np.max(sorted_vib[ 0 : int(np.round(inlier_fraction * flen)) ])

    # linear regression
    pt = np.polyfit( np.arange(0, flen), b[i:i + flen], 1)
    p[match_index] = pt[0]

    # under vibration threshold
    stable = max_diff <= dt
    # forward match # What's with the -1 and +1? I think this should be checking positions?
    forward_match = np.abs(p[match_index] - 1) < st or np.abs(p[match_index] + 1) < st

    # Nothing makes it through this filter, despite it working in the matlab version
    if stable and forward_match:
        matches[match_index] = pt[1] + pt[0] * 0.5 * flen

### Compare to ground truth ###

print("comparing to ground truth")
ground_truth = scipy.io.loadmat('GroundTruth_Eynsham_40meters.mat')['ground_truth']
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
    # '16' here is the consistent shift between the ground truth
    value_fit3 = value_fit2 - start_first + 1
    value_fit4 = value_fit3[ np.where(value_fit3 > 0)[0].astype(int) ]

    matrix_idx = ground_idx - start_second + 1
    ground_matrix[matrix_idx, value_fit4] = 1

for truth_idx in range(0, matches.size):

    ground_row = ground_truth[truth_idx+end_first, :]
    ground_row_idx = ground_row.toarray().flatten().nonzero()[0]

    # Maybe check if ground_row_idx is getting value that are not one?

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
