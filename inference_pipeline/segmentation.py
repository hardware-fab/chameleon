"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it), 
    Davide Zoni (davide.zoni@polimi.it)
"""

import numpy as np


def majorityFilter(array, kernel_size=None):
    """
    Apply a majority filter to a 1D array of integers.

    Parameters
    ----------
    array : np.ndarray
      The input array of interger. Value are limited from 0 to 2.
    kernel_size : int, optioanl
      The size of the window (default is None, which is equal to the length of the array).

    Returns
    -------
    The output array after applying the majority filter.
    """

    if kernel_size is None:
        value_counts = np.bincount(array)
        # Find the majority value (index of the maximum count)
        majority_value = np.argmax(value_counts)

        return np.full_like(array, majority_value)

    # Pad the array to handle edge cases
    pad_width = kernel_size // 2
    padded_array = np.pad(array, (pad_width, pad_width), mode='symmetric')

    output_array = np.zeros_like(array)

    for i in range(0, len(array)):
        window = padded_array[i:i + kernel_size]

        # Count the occurrences of each value
        value_counts = np.bincount(window)

        # Find the majority value (index of the maximum count)
        majority_value = np.argmax(value_counts)

        # Set the output element to the majority value
        output_array[i] = majority_value

    return output_array


def extractEdges(square_signal: np.ndarray) -> np.ndarray:
    """
    Get the starting sample for each CPs present in the square wave signal.

    Parameters
    ----------
    `square_signal` : np.ndarray
        The square wave signal.
    `min_distance` : int
        The minimum distance between two CPs.
    

    Returns
    -------
    The starting sample for each CPs present in the square wave signal.
    """

    starts = []
    ends = []

    if np.sum(square_signal[:10]) == 10:
        starts.append(1)

    for i in range(1, len(square_signal)):
        if square_signal[i] == 1 and square_signal[i - 1] != 1:
            starts.append(i+1)
        if (square_signal[i] == 2 and square_signal[i - 1] != 2) or (square_signal[i] == 1 and square_signal[i - 1] == 0):
            ends.append(i)

    return np.asarray(starts), np.asarray(ends)


def minCPsLenght(CPs: dict) -> int:
    '''
    Compute the minimum distance between two following CPs.

    Parameters
    ----------
    `CPs` : dict
        The starting sample for each CPs.

    Returns
    -------
    The minimum distance between two following CPs.
    '''

    # Handle empty list case
    if len(CPs['starts']) == 0 or len(CPs['ends']) == 0:
        return float('inf')
 
    distances = []
    for i in range(1, len(CPs['starts'])):
        distance = CPs['starts'][i] - CPs['starts'][i - 1]
        distances.append(distance)
    
    # Take the 100 smallest distances
    distances = sorted(distances)[:100]
    
    # Remove outliers
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_distances = [d for d in distances if lower_bound <= d <= upper_bound]
    
    # Return the mean of the remaining distances
    if filtered_distances:
        min_distance = np.mean(filtered_distances)
    else:
        min_distance = float('inf')
            
    return min_distance


def percentileCPsLenght(startCPs: np.ndarray, percentile: int = 5) -> int:
    '''
    Compute the percentile distance between two following CPs.

    Parameters
    ----------
    `startCPs` : array_like
        The starting sample for each CPs.
    `percentile` : int, optional
        The percentile to compute. Values must be between 0 and 100 (default is 5).

    Returns
    -------
    The percentile distance between two following CPs.
    '''

    # Handle empty list case
    if len(startCPs) == 0:
        return 0

    distances = []
    np.sort(startCPs)
    for i in range(1, len(startCPs)):
        distances.append(startCPs[i] - startCPs[i - 1])

    return np.percentile(distances, percentile)


def removeLastRound(classification, min_distance):
    '''
    Remove the last round of CPs from the classification output.

    Parameters
    ----------
    `classification` : np.ndarray
        The sliding windows classification output.
    `min_distance` : int
        The minimum distance between two CPs.

    Returns
    -------
    The updated classification output with the last round of CPs removed.
    '''

    min_distance = int(min_distance*0.35)
    idx = np.where(classification[:-min_distance] == 1)[0]
    
    for i in range(0, len(idx)):
        if majorityFilter(classification[idx[i]+1:idx[i]+min_distance+1])[0] == 2:
            classification[idx[i]] = 0

    return classification


def segment(classification: np.ndarray,
            major_filter_size: int,
            stride: int,
            avg_co_lenght: int) -> np.ndarray:
    '''
    Segment the classification output to obtain the start samples of the CPs.

    Parameters
    ----------
    `classification` : np.ndarray
        The sliding windows classification output.
    `major_filter_size` : int
        The initial size of the majority filter. It will be reduced during the segmentation.
    `stride` : int
        The stride used to slide the window in sliding windows classification.
    `avg_co_lenght` : int
        The initial minimum distance between two CPs. It will be refine during the segmentation.
        Rule of thumb: cipher average lenght.

    Returns
    -------
    The start samples of the CPs.
    '''
    CPs = {'starts': [], 'ends': []}
    offsets = []
    
    classifications_to_polish = [classification.copy()]
    
    min_distance = avg_co_lenght // stride
    
    while(major_filter_size>=1 and len(classifications_to_polish) > 0):
    
        polished_classifications = _polish(classifications_to_polish, major_filter_size, min_distance)

        first = len(CPs['ends'])==0 or len(CPs['starts'])==0
        newStartCPs, newEndCPs= _extract(polished_classifications, offsets, first)
        CPs = _insertNewCPs(CPs, newStartCPs, newEndCPs, min_distance, first)
        
        major_filter_size, min_distance, classifications_to_polish, offsets = _refine(classification, major_filter_size, CPs)
    
    #CPs = _finalizeSegmentation(classification, avg_co_lenght // stride, CPs)
    
    return CPs

def _insertNewCPs(CPs, newStartCPs, newEndCPs, min_distance, first):
    '''
    Insert the new starts and ends in the list of CPs.
    
    Parameters
    ----------
    `CPs` : np.ndarray
        The list of CPs.
    `newStartCPs` : np.ndarray
        The new startng CP's samples to insert.
    `newEndCPs` : np.ndarray
        The new ending CP's samples to insert.
    `min_distance` : int
        The minimum distance between two CPs.
    
    Returns
    -------
    The updated list of CPs.
    '''
    newInstert_start = list(set(newStartCPs) - set(CPs['starts']))
    newInstert_end = list(set(newEndCPs) - set(CPs['ends']))
    CPs['starts'].extend(newInstert_start)
    CPs['ends'].extend(newInstert_end)
    CPs['starts'] = sorted(CPs['starts'])
    CPs['ends'] = sorted(CPs['ends'])
    
    def removeClosest(list_, new_insert, min_distance):
        i = 1
        while i < len(list_):
            if list_[i] - list_[i - 1] < min_distance:
                if list_[i] in new_insert:
                    list_.pop(i)
                else:
                    list_.pop(i-1)
            else:
                i += 1
        return list_
    
    #if not first:
    CPs['starts'] = removeClosest(CPs['starts'], newInstert_start, min_distance)
    CPs['ends'] = removeClosest(CPs['ends'], newInstert_end, min_distance)
    
    return CPs
    

def _polish(classifications: list[np.ndarray],
            major_filter_size: int,
            min_distance: int) -> np.ndarray:
    '''
    Polish the classification output.
    It applies a majority filter and removes the last round of CPs.
    
    Parameters
    ----------
    `classification` : list[np.ndarray]
        The sliding windows classification output.
    `major_filter_size` : int
        The kernel size of the majority filter.
    `min_distance` : int
        The minimum distance between two CPs.

    Returns
    -------
    The filtered classification output.
    '''
    outs = []
    for classification in classifications:
        out = np.argmax(classification, axis=1)
        if major_filter_size > 1:
            out = majorityFilter(out, major_filter_size)
        
        out = removeLastRound(out, min_distance)
        
        outs.append(out)
    return outs

def _extract(classifications, offsets, first):
    '''
    Extract the starting sample for each CPs present in the classification output.
    
    Parameters
    ----------
    `classifications` : list[np.ndarray]
        The sliding windows classification output.
    `offsets` : list[int]
        The offset to apply to the starting sample.
    `first` : bool
        If True, the last edge is not detected.
    
    Returns
    -------
    The starting sample for each CPs present in the classification output.
    '''
    startCPs = []
    endCPs = []
    
    if first:
        offsets = [0] * len(classifications)
        
    for classification, offset in zip(classifications, offsets):
        newStarts, newEnds = extractEdges(classification)
        newStarts += offset
        newEnds += offset
        startCPs.extend(newStarts.tolist())
        endCPs.extend(newEnds.tolist())
        
    return startCPs, endCPs

def _refine(classification, major_filter_size, CPs):
    '''
    Refine the segmentation parameters to avoid false positive CPs.
    
    Parameters
    ----------
    `classifications` : np.ndarray
        The sliding windows classification output.
    `major_filter_size` : int
        The kernel size of the majority filter.
    `avg_co_lenght` : int
        The average distance between two CPs.
    `CPs` : np.ndarray
        The starting sample for each CPs.
    
    Returns
    -------
    The refined kernel size of the majority filter, the refined classification output and the refined offsets.
    '''
    scale = 0.8
    major_filter_size = _refineMajorFilterSize(major_filter_size)
    min_distance = int(minCPsLenght(CPs)*scale)
    
    sub_classifications = []
    offsets = []
     
    if len(CPs['starts']) == 0 or len(CPs['ends']) == 0:
        sub_classifications.append(classification)
        offsets.append(0)
        return major_filter_size, min_distance, sub_classifications, offsets
    
    # If can still be a CO at the beginning of the signal
    if CPs['starts'][0] > min_distance:
        sub_classifications.append(classification[: CPs['starts'][0]])
        offsets.append(0)
            
    # If can be two COs consecutives
    for i in range(1, len(CPs['starts'])):
        if CPs['starts'][i] - CPs['starts'][i - 1] > 2*min_distance:
            offset = int(min_distance*0.8)
            sub_classifications.append(classification[CPs['starts'][i-1]+offset: CPs['starts'][i]+1])
            offsets.append(CPs['starts'][i-1]+offset)

    # If can still be a CO at the end of the signal
    if len(classification) - CPs['starts'][-1] > 2*min_distance:
        offset = int(min_distance)
        sub_classifications.append(classification[CPs['starts'][-1]+offset:])
        offsets.append(CPs['starts'][-1]+offset)
     
    return major_filter_size, min_distance, sub_classifications, offsets

def _refineMajorFilterSize(major_filter_size):
    if major_filter_size > 299:
        return major_filter_size // 2
    elif major_filter_size > 9:
        return major_filter_size // 10
    elif major_filter_size > 1:
        return 1
    elif major_filter_size <= 1:
        return 0
