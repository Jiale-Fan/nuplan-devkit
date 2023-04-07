import os
from pathlib import Path
import hydra
from nuplan.planning.script.run_simulation import main as main_simulation
from nuplan.planning.script.run_nuboard import main as main_nuboard
from omegaconf import DictConfig
import warnings


def simulate(sim_dict: dict) -> str:
    # Location of path with all simulation configs
    CONFIG_PATH = sim_dict['CONFIG_PATH']
    CONFIG_NAME = sim_dict['CONFIG_NAME']

    # Select the planner and simulation challenge
    PLANNER = sim_dict['PLANNER']  # [simple_planner, ml_planner]
    CHALLENGE = sim_dict['CHALLENGE']  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    DATASET_PARAMS = sim_dict['DATASET_PARAMS']
    
    # Name of the experiment
    EXPERIMENT = sim_dict['EXPERIMENT']
    
    # add save directory
    SAVE_DIR = sim_dict['SAVE_DIR']
    
    simulation_folder = ''
    
    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)
    
    if PLANNER == 'simple_planner':
        # Compose the configuration
        cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
            f'experiment_name={EXPERIMENT}',
            f'group={SAVE_DIR}',
            f'planner={PLANNER}',
            f'+simulation={CHALLENGE}',
            *DATASET_PARAMS,
        ])
        
        # Run the simulation loop
        main_simulation(cfg)
    
        # Simulation folder for visualization in nuBoard
        simulation_folder = cfg.output_dir
        
    elif PLANNER == 'ml_planner':
        
        MODEL = sim_dict['MODEL']
        
        # LOG_DIR = str(Path(SAVE_DIR) / '..' / 'training' / EXPERIMENT / MODEL)
        LOG_DIR = str(Path(SAVE_DIR) / '..' / EXPERIMENT / MODEL)
        print(LOG_DIR)

        # Get the checkpoint of the trained model
        last_experiment = sorted(os.listdir(LOG_DIR))[-1]
        
        i = -1
        train_experiment_dir = sorted(Path(LOG_DIR).iterdir())[i]  # get last experiment
        #################################################################################
        # while not (train_experiment_dir / 'checkpoints').exists():
        #     i -= 1
        #     train_experiment_dir = sorted(Path(LOG_DIR).iterdir())[i]  # get last experiment
        #     if i == -10: break
        # checkpoint = sorted((train_experiment_dir / 'checkpoints').iterdir())[-1]  # get last checkpoint
        ###################################################################################
        while not (train_experiment_dir / 'best_model').exists():
            i -= 1
            train_experiment_dir = sorted(Path(LOG_DIR).iterdir())[i]  # get last experiment
            if i == -10: break
        checkpoint = sorted((train_experiment_dir / 'best_model').iterdir())[-1]



        MODEL_PATH = str(checkpoint).replace("=", "\=")

        print("Model path is: ")
        print(MODEL_PATH)

        
        # Compose the configuration
        cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
            f'experiment_name={EXPERIMENT}',
            f'group={SAVE_DIR}',
            f'planner={PLANNER}',
            f'model={MODEL}',
            'planner.ml_planner.model_config=${model}',  # hydra notation to select model config
            f'planner.ml_planner.checkpoint_path={MODEL_PATH}',  # this path can be replaced by the checkpoint of the model trained in the previous section
            f'+simulation={CHALLENGE}',
            *DATASET_PARAMS,
        ])
    
        # Run the simulation loop
        main_simulation(cfg)
    
        # Simulation folder for visualization in nuBoard
        simulation_folder = cfg.output_dir
    
    return simulation_folder


def open_nuboard(simulation_folders: list[str]) -> None:
    # Location of path with all nuBoard configs
    CONFIG_PATH = '../nuplan/planning/script/config/nuboard'
    CONFIG_NAME = 'default_nuboard'

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)

    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        'scenario_builder=nuplan_mini',  # set the database (same as simulation) used to fetch data for visualization
        f'simulation_path={simulation_folders}',  # nuboard file path(s), if left empty the user can open the file inside nuBoard
    ])
    
    # Run nuBoard
    main_nuboard(cfg)
    
    
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sim_dicts = []
    
    sim_dicts.append(
        dict(
            # Location of path with all simulation configs
            CONFIG_PATH = '../nuplan/planning/script/config/simulation',
            CONFIG_NAME = 'default_simulation',

            # Select the planner and simulation challenge
            PLANNER = 'ml_planner',  # [simple_planner, ml_planner]
            CHALLENGE = 'open_loop_boxes',  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
            DATASET_PARAMS = [
                'scenario_builder=nuplan_mini',  # use nuplan mini database
                'scenario_filter=all_scenarios',  # initially select all scenarios in the database
                'scenario_filter.scenario_types=[near_multiple_vehicles, on_pickup_dropoff, starting_unprotected_cross_turn, high_magnitude_jerk]',  # select scenario types
                'scenario_filter.num_scenarios_per_type=10',  # use 10 scenarios per scenario type
            ],
        
            # Name of the experiment
            EXPERIMENT = 'autobots_experiment',
            
            # add save directory
            SAVE_DIR = '/data1/nuplan/jiale/exp/simulation',
            
            # for ML Planner only
            MODEL = 'autobots_model'
        )
        
    )
   
    
    simulation_folders = []
    for sim_dict in sim_dicts:
        simulation_folder = simulate(sim_dict)
        simulation_folders.append(simulation_folder)
        
    open_nuboard(simulation_folders)
    

    """
    conda activate nuplan
cd ~/Documents/master/nuplan-devkit
python ./scripts/simulate_autobots.py 
    """