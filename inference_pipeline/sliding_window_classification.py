"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it), 
    Davide Zoni (davide.zoni@polimi.it)
"""

import os
from tqdm.auto import tqdm
import torch
import numpy as np
from math import ceil
from CNN.utils import (parse_arguments, get_neptune_run,
                       get_experiment_config_dir)
import CNN.modules as modules
from CNN.build_dataset_chameleon import highpass
import h5py


def _cutSubWindows(trace, window_size, stride, batch_size):
    windows = []
    for i in range(trace.shape[0] // stride):
        start = i * stride
        end = min(window_size + start, trace.shape[0])
        # Yield full or partial batches
        if len(windows) == batch_size or end == trace.shape[0]:
            yield windows
            windows = []

        # Only append non-empty windows
        if end > start:
            windows.append(trace[start:end])

    # Yield the final single-trace batch
    if windows:
        yield windows


def _classify(trace, module, window_size,
              device, stride, batch_size=2048):

    classified_points = []
    for windows_batch in tqdm(_cutSubWindows(trace, window_size, stride, batch_size), 
                              leave=False,
                              total=ceil(trace.shape[0] / (stride*batch_size))):
        windows_batch = torch.from_numpy(np.array(windows_batch)).to(device)
        windows_batch = windows_batch.reshape(windows_batch.shape[0], -1)

        with torch.no_grad():
            y_hat = module(windows_batch)
            y_hat = torch.nn.functional.softmax(y_hat, 1)
            classified_points.append(y_hat.detach())

    return torch.cat(classified_points, dim=0).cpu().data.numpy()


def _slidingWindowClassify(module, traces, device, window_size, stride, batch_size=2048):

    classified_points = []
    for trace in tqdm(traces, colour='green'):
        trace = highpass(trace, 0.001)
        trace = (trace - np.mean(trace, axis=0)) / np.std(trace, axis=0)
        ret = _classify(trace, module, window_size, device, stride, batch_size)
        classified_points.append(ret)

    classified_points = np.asarray(classified_points)

    return classified_points


def getModule(SID: str,
              neptune_config: str = 'CNN/configs/common/neptune_configs.yaml') -> torch.nn.Module:
    """
    Get the best model from a Neptune Run and return the corresponding module.

    Parameters
    ----------
    `SID` : str
        The Neptune Run ID.
    `neptune_config` : str, optional
        The path to the Neptune configuration file (default is 'CNN/configs/common/neptune_configs.yaml').

    Returns
    -------
    The best model from the Neptune Run.
    """

    # Get Neptune Run (by SID)
    df = get_neptune_run(neptune_config, SID)

    # Get experiment name
    exp_name = df['sys/name'].iloc[0]

    # Get best model path
    best_model_ckpt = df['experiment/model/best_model_path'].iloc[0]

    # Get config dir
    config_dir = get_experiment_config_dir(best_model_ckpt, exp_name)

    _, module_config, __ = parse_arguments(config_dir)

    # Build Model
    # -----------
    module_name = module_config['module']['name']
    module_config = module_config['module']['config']
    module_class = getattr(modules, module_name)
    module = module_class.load_from_checkpoint(best_model_ckpt,
                                               module_config=module_config)

    return module


def classifyTrace(trace_file: str,
                  module: torch.nn.Module,
                  stride: int,
                  window_size: int,
                  batch_size: int = 2048,
                  gpu: int = 0) -> np.ndarray:
    """
    Classify a side-channel trace using a sliding-windows approach.
    Stride and window size are configurable.

    Parameters
    ----------
    `trace_file` : str
        The path to the trace file to classify.
    `module` : torch.nn.Module
        The CNN module to use for classification.
    `stride` : int
        The stride to use for the sliding window.
    `window_size` : int
        The size of the sliding window.
    `gpu` : int, optional
        The GPU to use for classification (default is 0).
        0 means the first GPU, 1 means the second GPU, and so on.

    Returns
    -------
    A classification score for each winodw in the trace.
    """
    # Get Device
    device = torch.device(
        f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # Set Module
    module.to(device)
    module.eval()

    traces = _dataLoader(trace_file)
    
    segmentation = _slidingWindowClassify(module, traces, device,
                                          window_size, stride, batch_size)
    return segmentation


def saveClassification(segmentation: np.ndarray,
                       output_file: str) -> None:
    """
    Save the segmentation to a file.

    Parameters
    ----------
    `segmentation` : np.ndarray
        The segmentation to save.
    `output_file` : str
        The file where the segmentation will be saved.
    """

    np.save(output_file, segmentation)

def _dataLoader(chunk_file):
    with h5py.File(chunk_file, 'r', libver='latest') as hf_chunk:
        chunk_len = len(hf_chunk['metadata/ciphers/'].keys())
        for n in range(0, chunk_len):
            traces = hf_chunk[f'data/traces/trace_{n}']
            yield traces[:]