import os
from pathlib import Path
import hydra
from nuplan.planning.script.run_training import main as main_train
from omegaconf import DictConfig
import tempfile
import pickle


file_name="./project_records/laneGCN_input_sample.obj"

with open(file_name, 'rb') as file:
    datainput=pickle.load(file)
    print(f'Object successfully loaded to "{file_name}"')

from nuplan.planning.training.preprocessing.feature_builders.autobots_feature_builder import AutobotsMapFeatureBuilder, AutobotsAgentsFeatureBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

def testMap():
    fb=AutobotsMapFeatureBuilder(1.0)

    vec_map=datainput[0]['vector_map']

    ts=fb.VectorMapToAutobotsMapTensor(vec_map)

    print(ts)

def testAgents():
    fb=AutobotsAgentsFeatureBuilder(TrajectorySampling(num_poses=4, time_horizon=1.5))

    ag=datainput[0]['agents']

    ts=fb.AgentsToAutobotsAgentsTensor(ag)

    print(ts)

testAgents()
# testMap()