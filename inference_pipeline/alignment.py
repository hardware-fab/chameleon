"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it), 
    Davide Zoni (davide.zoni@polimi.it)
"""

import numpy as np

def alignCps(trace: np.ndarray,
             startCPs: list[np.ndarray],
             stride: int):
    """
    Locate the CPs in the trace and align them in time.
    Trace and stride are the same used for the sliding window classification.
    
    Parameters
    ----------
    `trace` : np.ndarray
        The trace to align. To be set as the trace used for the sliding window classification.
    `startCPs` : list[np.ndarray]
        The starting sample for each CPs present in the trace.
    `stride` : int
        The stride used to align the CPs. To be set as the stride used for the sliding window.
    
    Returns
    -------
    The aligned CPs, i.e., a matrix with a row for each CO.
    """
    algned_cos = []

    for n_trace, startSample in enumerate(startCPs):
        cos = []
        for p_id in range(len(startSample)):
            start = startSample[p_id] * stride
            end = trace.shape[1] - 1 if p_id == len(startSample)-1 else startSample[p_id + 1] * stride
            cos.append(trace[n_trace, start:end])
        algned_cos.extend(np.asarray(cos, dtype=object))
        
    return algned_cos

def saveCps(cos: list,
            output_file: str):
    """
    Save the aligned CPs to a file.
    
    Parameters
    ----------
    `cos` : list
        The aligned CPs.
    `output_file` : str
        The file to save the aligned CPs.
    """
    
    # CPs have diffent length, so we need to save them as object
    np.save(output_file, np.array(cos, dtype=object))
    
def padCps(cos_to_pad):
    """
    Pad the CPs to the same length.
    Since the CPs have different length, we need to padd them to the same length.
    We choose the minimum length among all the CPs.
    
    Parameters
    ----------
    `cos_to_pad` : np.ndarray
        The aligned CPs.
    
    Returns
    -------
    The aligned CPs with the same length.
    """
    
    min_len = len(min(cos_to_pad, key = lambda x: len(x)))
    new_seg = [co[:min_len] for co in cos_to_pad]
    return np.array(new_seg, dtype=np.float32)