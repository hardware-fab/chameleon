"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it), 
    Davide Zoni (davide.zoni@polimi.it)
"""

import numpy as np
import h5py


def errorCount(gt, predictions, stride):
    '''
    Returns the number of false positives and false negatives between the ground truth and the predictions.

    Parameters
    ----------
    gt : numpy.ndarray
        The ground truth CPs.
    predictions : numpy.ndarray
        The predictions.
    stride : int
        The stride used during Sliding Window Classification.

    Returns
    -------
    int
        The number of false positives, i.e., the number of CPs found in the predictions but not in the ground truth. 
    int
        The number of false negatives, i.e., the number of CPs found in the ground truth but not in the predictions.
    '''
    fp = len(falsePositives(gt, predictions, stride))
    fn = len(falseNegatives(gt, predictions, stride))
    return fp, fn


def errorRate(gt, predictions, stride):
    '''
    Returns the error rate between the ground truth and the predictions.
    The error rate is defined as the number of errors divided by the number of CPs in the ground truth.
    '''
    fp, fn = errorCount(gt, predictions, stride)
    return fp / len(gt), fn / len(gt)


def _findClosestGt(gt, predictions):
    '''
    Returns the closest ground truth CPs to the predictions.
    '''
    closest_gt = []
    for pred in predictions:
        distances = np.linalg.norm(gt - pred.reshape(1, -1), axis=0)
        closest_gt.append(gt[np.argmin(distances)])

    return closest_gt


def errorDistance(gt, predictions, stride, relative='gt'):
    '''
    Returns the error distance between the ground truth and the predictions.
    The error distance is defined as the Euclidean distance between the closest ground truth CO and the prediction.

    Parameters
    ----------
    gt : numpy.ndarray
        The ground truth CPs.
    predictions : numpy.ndarray
        The predictions.
    stride : int
        The stride used during Sliding Window Classification.
    relative : str, optional
        The reference frame to compute the error distance (default is 'gt').
        If 'gt', the error distance is computed in the ground truth reference frame.
        If 'pred', the error distance is computed in the predictions reference frame.

    Returns
    -------
    list
        The error distances for each prediction from the ground truth.
    '''
    if relative == 'pred':
        gt = gt//stride
    elif relative == 'gt':
        predictions = predictions*stride
    else:
        raise ValueError("relative must be 'gt' or 'pred'")

    closest_gt = _findClosestGt(gt, predictions)
    errors = []
    for pred, gt_ in zip(predictions, closest_gt):
        error = gt_ - pred
        errors.append(error)
    return errors


def __nonUniqueClosestGt(gt):
    for i, x in enumerate(gt):
        if gt[:i].count(x) == 0 and gt[i+1:].count(x) > 0:
            yield i, gt.count(x)


def falsePositives(gt, predictions, stride):
    '''
    Returns the number of false positive corresponding to non-true CPs.
    An error is defined as a CO found in the prediction but not in the ground truth.

    Parameters
    ----------
    gt : numpy.ndarray
        The ground truth CPs.
    predictions : numpy.ndarray
        The predictions.
    stride : int
        The stride used during Sliding Window Classification.

    Returns
    -------
    The false positives.
    '''

    false_positives = []
    closest_gt = _findClosestGt(gt//stride, predictions)
    errors = np.abs(errorDistance(gt, predictions,
                    stride=stride, relative='pred'))

    non_unique_closest_gt = __nonUniqueClosestGt(closest_gt)

    for i in non_unique_closest_gt:
        idx, num = i
        min_error = min(errors[idx:idx+num])
        for j in range(num):
            if errors[idx+j] > min_error:
                false_positives.append(predictions[idx+j])

    return false_positives


def falseNegatives(gt, predictions, stride):
    '''
    Returns the number of false negative corresponding to true CPs.
    An error is defined as a CO found in the ground truth but not in the predictions.

    Parameters
    ----------
    gt : numpy.ndarray
        The ground truth CPs.
    predictions : numpy.ndarray
        The predictions.
    stride : int
        The stride used during Sliding Window Classification.

    Returns
    -------
    The false negatives.
    '''

    false_negatives = []
    closest_gt = _findClosestGt(gt//stride, predictions)

    false_negatives = np.setdiff1d(gt//stride, closest_gt)

    return false_negatives

def IoU(gt_starts, gt_ends, predictions, stride):
    '''
    Returns the Intersection over Union (IoU) between the ground truth and the predictions.
    The IoU is defined as the intersection between the ground truth and the predictions divided by the union of the ground truth and the predictions.

    Parameters
    ----------
    gt_starts : numpy.ndarray
        The ground truth start CPs.
    gt_ends : numpy.ndarray
        The ground truth end CPs.
    predictions : numpy.ndarray
        The predictions.

    Returns
    -------
    float
        The Intersection over Union (IoU).
    '''
    gt_starts, gt_ends, predictions = _removeErrors(gt_starts, gt_ends, predictions, stride)
    assert len(gt_starts) == len(predictions)
    assert len(gt_ends) == len(predictions)
    
    iou = []
    for start, end, prediction in zip(gt_starts, gt_ends, predictions):
        union = end - min(start, prediction*stride)
        intersection = end - max(start, prediction*stride)
        iou.append(intersection / union)
        
    return np.mean(iou)

def _removeErrors(gt_starts, gt_ends, predictions, stride):
    false_positives = falsePositives(gt_starts, predictions, stride)
    predictions = np.setdiff1d(predictions, false_positives)
    
    false_negatives = falseNegatives(gt_starts, predictions, stride)
    removed_indices = np.where(np.isin(gt_starts//stride, false_negatives))[0]
    gt_starts = np.delete(gt_starts, removed_indices)
    gt_ends = np.delete(gt_ends, removed_indices)
    return gt_starts,gt_ends,predictions

def loaderGt(chunk_file):
    with h5py.File(chunk_file, 'r', libver='latest') as hf_chunk:
        chunk_len = len(hf_chunk['metadata/ciphers/'].keys())
        for n in range(0, chunk_len):
            labels = hf_chunk[f'metadata/pinpoints/pinpoints_{n}']
            yield labels[:]
