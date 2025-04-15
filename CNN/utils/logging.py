"""
Authors : 
    Giuseppe Chiari (giuseppe.chiari@polimi.it),
    Davide Galli (davide.galli@polimi.it), 
    Davide Zoni (davide.zoni@polimi.it)
"""

import os
import shutil
from omegaconf import OmegaConf

from datetime import datetime
import neptune
import logging

from pytorch_lightning.loggers import NeptuneLogger

class _FilterCallback(logging.Filterer):
    def filter(self, record: logging.LogRecord):
        return not (
            record.name == "neptune"
            and record.getMessage().startswith(
                "Error occurred during asynchronous operation processing: X-coordinates (step) must be strictly increasing for series attribute"
            )
        )

neptune.internal.operation_processors.async_operation_processor.logger.addFilter(
    _FilterCallback()
)

def build_neptune_logger(exp_name, tags, neptune_config_file, SID=None):
    neptune_config = OmegaConf.to_object(
        OmegaConf.load(neptune_config_file))

    user = neptune_config['user']
    token = neptune_config['token']
    project = neptune_config['project']
    id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    kwargs = {}
    kwargs['prefix'] = 'experiment'
    kwargs['project'] = f'{user}/{project}'
    kwargs['name'] = f'{exp_name}_{id}'
    kwargs['description'] = ''
    kwargs['tags'] = tags

    kwargs['source_files'] = [
        'CNN/*'
    ]

    if SID is not None:
        df = get_neptune_run(neptune_config_file, SID)
        if not df.empty:
            run = neptune.Run(
                api_token=token, project=kwargs['project'] , with_id=SID
            )
            neptune_logger = NeptuneLogger(run=run, prefix='experiment')
            name = df['sys/name'].values[0]
        else:
            raise Exception("No run found with the given SID")
    else:
        neptune_logger = NeptuneLogger(api_key=token, **kwargs)
        name = kwargs['name']
    
    return neptune_logger, name


def get_neptune_run(neptune_config_file, SID):
    neptune_config = OmegaConf.to_object(
        OmegaConf.load(neptune_config_file))

    user = neptune_config['user']
    token = neptune_config['token']
    project = neptune_config['project']

    project = neptune.init_project(
        api_token=token, project=f'{user}/{project}', mode='read-only')
    df = project.fetch_runs_table().to_pandas()

    df = df[df['sys/id'] == SID]

    return df

def get_checkpoint(neptune_config_file, SID):
    df = get_neptune_run(neptune_config_file, SID)
    if not df.empty:
        checkpoint = df['experiment/model/best_model_path'].values[0]
    else:
        raise Exception("No run found with the given SID")
    return checkpoint

def init_experiment_dir(exp_config, exp_name):
    exp_dir = os.path.join(exp_config['log_dir'], exp_name)
    print(exp_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    #else:
    #    raise Exception("Experiment folder already exists.")
    return exp_dir


def save_experiment_configs(exp_dir, original_config_dir):
    config_dir = os.path.basename(os.path.normpath(original_config_dir))
    shutil.copytree(
        original_config_dir,
        os.path.join(exp_dir, 'configs', config_dir))
