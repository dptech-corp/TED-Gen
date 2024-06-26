import numpy as np
from math import cos, sin, tan,degrees, radians
def shift_trans(x,y):
    all_targets = np.array([[x,y,1.]])

    all_targets_symmetric = all_targets.copy()
    all_targets_symmetric[:, 1] = -all_targets_symmetric[:, 1]

    all_targets_filp = all_targets_symmetric.copy()
    all_targets_filp[:, :2] = (0, 0) - (all_targets_symmetric[:, :2] - (0, 0))
    points = np.concatenate([all_targets_filp,all_targets_symmetric])

    condition_indices = np.where(points[:, 0]<0)

    shift_value = 6.42
    points[condition_indices, 0] += shift_value 
    condition_indices = np.where(points[:, 1]<0)
    points[condition_indices, 0] += -3.21
    points[condition_indices, 1] += 5.71
    condition_indices = np.where(1.74*points[:,0]+points[:, 1]<0)
    points[condition_indices, 0] += 6.42
    condition_indices2 = np.where(1.74*points[:,0]+points[:, 1]>11.21)
    points[condition_indices2, 0] += -6.42

    index = []
    for i in range(points.shape[0]):
        if (5.71/(3.27+6.42))*points[i][0] + points[i][1] > (5.71/(3.27+6.42))*6.42:
            index.append(i)

    points = np.delete(points,index,axis=0)

    angle_in_degrees1 = 119.8- 90
    angle_in_radians1 = radians(angle_in_degrees1)
    angle_in_degrees2 = 60.2
    angle_in_radians2 = radians(angle_in_degrees2)
    new_x = [p[0]+p[1]*tan(angle_in_radians1) for p in points]
    new_y = [p[1]/cos(angle_in_radians1) for p in points]
    target_new = np.stack([new_x, new_y],axis=1)

    return target_new[0]