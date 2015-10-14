from constants import *

def spatial_check(conf_matrix, hypothesis):
    if hypothesis - DEE > 0:
        start = hypothesis - DEE
    else:
        start = DEE

    res = True
    for hyp in range(start, hypothesis):
        res = np.abs(conf_matrix[hyp - 1] - conf_matrix[hyp]) <= EPSILON

        if not res:
            break
    return res