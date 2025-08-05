#!/usr/bin/env python3
"""
Main entry point for running the definitive SFT v2.1 experiment.
"""
from config.config import ExperimentConfigs
from experiments.experiment_runner import SFT_Relational_Experiment

def main():
    """
    Selects the configuration and runs the SFT v2 position-aware experiment.
    """
    # Use the new, definitive config for the position-aware model
    config = ExperimentConfigs.position_aware_svo()
    
    # Initialize and run the experiment
    experiment = SFT_Relational_Experiment(config)
    experiment.run()

if __name__ == "__main__":
    main()