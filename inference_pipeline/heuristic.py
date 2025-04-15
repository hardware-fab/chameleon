"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it), 
    Davide Zoni (davide.zoni@polimi.it)
"""
from .segmentation import majorityFilter
import numpy as np


def removeFalsePositives_basic(COstarts: list, COends: list) -> tuple:
    '''
    Remove false positives from identified COs.
    This is a basic and simple implemantation of the algorithm, for 
    a more robust implemention ses "removeFalsePositives()"

    Parameters
    -----------
    COstarts : list
        List of start points of the COs.
    COends : list
        List of end points of the COs.

    Returns
    -------
    list
        List of start points of the COs after removing false positives.
    list
        List of end points of the COs after removing false positives.
    '''

    starts, ends = [], []
    i, j = 0, 0
    while i < len(COstarts)-1 and j < len(COends)-1:
        starts.append(COstarts[i])
        if COstarts[i] < COends[j] and \
                COstarts[i+1] >= COends[j]:
            i += 1
        else:
            i += 2
        ends.append(COends[j])
        if COstarts[i] >= COends[j] and \
                COstarts[i] < COends[j+1]:
            j += 1
        else:
            j += 2

    starts.extend(COstarts[i:])
    ends.extend(COends[j:])
    return starts, ends


def removeFalsePositives(COstarts: list, COends: list, classification: np.ndarray) -> tuple:
    '''
    Remove false positives from identified COs.
    This is a more robust implemantion of the algorithm, for
    a more basic implemention ses "removeFalsePositives_basic()"

    Parameters
    -----------
    COstarts : list
        List of start points of the COs.
    COends : list
        List of end points of the COs.
    classification : numpy.ndarray
        The classification output of the side-channel trace.

    Returns
    -------
    list
        List of start points of the COs after removing false positives.
    list
        List of end points of the COs after removing false positives.
    '''

    starts, ends = [], []
    min_distance = _minCP(COstarts)*0.9
    i, j = 0, 0
    while i < len(COstarts)-1 and j < len(COends)-1:
        starts.append(COstarts[i])
        if COstarts[i] < COends[j] and \
           COstarts[i+1] >= COends[j]:
            i += 1
            if majorityFilter(np.argmax(classification[COstarts[i-1]:COends[j]], axis=1))[1] != 0:
                starts.pop()
        else:
            if COends[j] - COstarts[i+1] > min_distance:
                starts.remove(COstarts[i])
                starts.append(COstarts[i+1])
            i += 2
        ends.append(COends[j])
        if COstarts[i] >= COends[j] and \
                COstarts[i] < COends[j+1]:
            j += 1
            if majorityFilter(np.argmax(classification[COstarts[i-1]:COends[j-1]], axis=1))[1] != 0:
                ends.pop()
        else:
            j += 2

    starts.extend(COstarts[i:])
    ends.extend(COends[j:])
    return starts, ends


def removeFalseNegatives(COstarts: list, COends: list, classification: np.ndarray) -> tuple:
    '''
    Remove false negatives from identified COs.

    Parameters
    -----------
    COstarts : list
        List of start points of the COs.
    COends : list
        List of end points of the COs.
    classification : numpy.ndarray
        The classification output of the side-channel trace.

    Returns
    -------
    list
        List of start points of the COs after removing false negatives.
    list
        List of end points of the COs after removing false negatives.
    '''

    i, j = 1, 0
    min_distance = _minCP(COstarts)*0.9

    while i < len(COstarts)-1 and j < len(COends)-1:
        if COstarts[i]-COends[j] > min_distance and \
                majorityFilter(np.argmax(classification[COends[j]:COstarts[i]], axis=1))[1] != 2:
            newStart = COends[j]+1
            newEnd = COstarts[i]-1
            COstarts.append(newStart)
            COends.append(newEnd)
        i += 1
        j += 1
    COstarts, COends = sorted(COstarts), sorted(COends)
    if COstarts[-1] > COends[-1]:
        COends.append(len(classification))
    return COstarts, COends


def _minCP(starts) -> int:

    distances = []
    for i in range(1, len(starts)):
        distance = starts[i] - starts[i - 1]
        distances.append(distance)

    # Take the 100 smallest distances
    distances = sorted(distances)[:100]

    # Remove outliers
    filtered_distances = _removeOutliers(distances)

    # Return the mean of the remaining distances
    if filtered_distances:
        min_distance = np.mean(filtered_distances)
    else:
        min_distance = float('inf')

    return min_distance


def _removeOutliers(distances):
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_distances = [
        d for d in distances if lower_bound <= d <= upper_bound]
    return filtered_distances
