import os
from pathlib import Path
import hydra
from nuplan.planning.script.run_training import main as main_train
from omegaconf import DictConfig
import tempfile


def train(sim_dict: dict) -> str:
    # Location of path with all simulation configs
    CONFIG_PATH = sim_dict['CONFIG_PATH']
    CONFIG_NAME = sim_dict['CONFIG_NAME']
    
    # add save directory
    SAVE_DIR = sim_dict['SAVE_DIR']
    # Name of the experiment
    EXPERIMENT = sim_dict['EXPERIMENT']
    JOB_NAME = sim_dict['JOB_NAME']
    LOG_DIR = str(Path(SAVE_DIR) / EXPERIMENT / JOB_NAME)
    print(LOG_DIR)

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)
    
    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'group={str(SAVE_DIR)}',
        f'cache.cache_path={str(SAVE_DIR)}/cache',
        f'experiment_name={EXPERIMENT}',
        f'job_name={JOB_NAME}',
        'py_func=train',
        '+training=training_raster_model',  # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
        'scenario_builder=nuplan_mini',  # use nuplan mini database
        'scenario_filter.limit_total_scenarios=500',  # Choose 500 scenarios to train with
        'lightning.trainer.params.accelerator=ddp_spawn',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
        'lightning.trainer.params.max_epochs=10',
        'data_loader.params.batch_size=8',
        'data_loader.params.num_workers=8',
    ])
    
    # Run the training loop, optionally inspect training artifacts through tensorboard (above cell)
    main_train(cfg)
    
    
if __name__ == '__main__':
    train_dicts = []
    # Raster Model
    # train_dicts.append(
    #     dict(
    #         # Location of path with all simulation configs
    #         CONFIG_PATH = '../nuplan/planning/script/config/training',
    #         CONFIG_NAME = 'default_training',
        
    #         # Name of the experiment
    #         EXPERIMENT = 'raster_experiment',
    #         JOB_NAME = 'raster_model',
    #         # add save directory
    #         SAVE_DIR = '/data1/nuplan/exp/exp/training'
    #     )
    # )
    # Vector Model
    train_dicts.append(
        dict(
            # Location of path with all simulation configs
            # CONFIG_PATH = '../nuplan/planning/script/experiments/training',
            # CONFIG_NAME = 'training_vector_model',
            # CONFIG_PATH = '../nuplan/planning/script/config/common/model',
            # CONFIG_NAME = 'vector_model',
            CONFIG_PATH = '../nuplan/planning/script/config/training',
            CONFIG_NAME = 'default_training',
        
            # Name of the experiment
            EXPERIMENT = 'vector_experiment',
            JOB_NAME = 'vector_model',
            # add save directory
            SAVE_DIR = '/data1/nuplan/exp/exp/training'
        )
    )
    
    for train_dict in train_dicts:
        train(train_dict)