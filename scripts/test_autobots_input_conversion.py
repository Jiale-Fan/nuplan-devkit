import os
from pathlib import Path
import hydra
from nuplan.planning.script.run_training import main as main_train
from omegaconf import DictConfig
import tempfile
import pickle
from nuplan.planning.training.preprocessing.feature_builders.autobots_feature_builder import AutobotsMapFeatureBuilder, AutobotsAgentsFeatureBuilder, AutobotsTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.features.autobots_feature_conversion import VectorMapToAutobotsMapTensor, AgentsToAutobotsAgentsTensor, TrajectoryToAutobotsEgoin

# file_name="./project_records/laneGCN_input_sample.obj" # one batch with batch size 1
file_name="./project_records/laneGCN_input_sample_batch.pkl" # one batch with batch size 2

with open(file_name, 'rb') as file:
    datainput=pickle.load(file)
    print(f'Object successfully loaded to "{file_name}"')



def testMap():

    vec_map=datainput[0]['vector_map']

    ts=VectorMapToAutobotsMapTensor(vec_map)

    print(ts)

def testAgents():

    ag=datainput[0]['agents']

    ts=AgentsToAutobotsAgentsTensor(ag)

    print(ts)

def testTrajectory():
    # fb=AutobotsTargetBuilder(TrajectorySampling(num_poses=10, time_horizon=5.0))

    ag=datainput[1]['trajectory']

    ts=TrajectoryToAutobotsEgoin(ag)

    print(ts)

testTrajectory()
# testAgents()
# testMap()