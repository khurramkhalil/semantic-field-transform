#!/usr/bin/env python3
"""
Main entry point for running the definitive SFT v2 experiment.
"""
from config.config import ExperimentConfigs
from experiments.experiment_runner import SFT_Relational_Experiment

def main():
    """
    Selects the configuration and runs the SFT v2 relational experiment.
    """
    # Get config for the relational prediction test
    config = ExperimentConfigs.relational_prediction()
    
    # Initialize and run the experiment
    experiment = SFT_Relational_Experiment(config)
    experiment.run()

if __name__ == "__main__":
    main()