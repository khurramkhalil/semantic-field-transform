#!/usr/bin/env python3
"""
Configuration file for SFT experiments
Centralized configuration management for research flexibility
"""

import torch


class SFTConfig:
    """
    Configuration class for SFT experiments
    Easily modifiable for different research directions
    """
    
    # Model Architecture
    SEMANTIC_DIM = 128
    FIELD_RESOLUTION = 128
    N_LAYERS = 3
    N_CLASSES = 2
    
    # Field Components
    LOCALITY_WIDTH = 0.02
    KINETIC_STRENGTH = 0.1
    POTENTIAL_STRENGTH = 0.05
    
    # Training
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01
    BATCH_SIZE = 8
    EPOCHS = 15
    
    # Data
    MAX_LENGTH = 64
    TEST_SPLIT = 0.2
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Visualization
    FIGURE_DPI = 300
    FIGURE_SIZE = (12, 8)
    
    @classmethod
    def create_model_config(cls):
        """Create model configuration dictionary"""
        return {
            'semantic_dim': cls.SEMANTIC_DIM,
            'field_resolution': cls.FIELD_RESOLUTION,
            'n_layers': cls.N_LAYERS,
            'n_classes': cls.N_CLASSES
        }
    
    @classmethod
    def create_training_config(cls):
        """Create training configuration dictionary"""
        return {
            'lr': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'batch_size': cls.BATCH_SIZE,
            'epochs': cls.EPOCHS,
            'device': cls.DEVICE
        }
    
    @classmethod
    def create_data_config(cls):
        """Create data configuration dictionary"""
        return {
            'max_length': cls.MAX_LENGTH,
            'test_split': cls.TEST_SPLIT,
            'batch_size': cls.BATCH_SIZE
        }


# Predefined experiment configurations
class ExperimentConfigs:
    """
    Predefined configurations for different experiments
    """
    
    @staticmethod
    def quick_test():
        """Configuration for quick testing"""
        config = SFTConfig()
        config.SEMANTIC_DIM = 64
        config.FIELD_RESOLUTION = 64
        config.N_LAYERS = 2
        config.EPOCHS = 5
        config.BATCH_SIZE = 4
        return config
    
    @staticmethod
    def full_experiment():
        """Configuration for full experiment"""
        return SFTConfig()
    
    @staticmethod
    def high_resolution():
        """Configuration for high-resolution field experiments"""
        config = SFTConfig()
        config.SEMANTIC_DIM = 256
        config.FIELD_RESOLUTION = 512
        config.N_LAYERS = 5
        config.EPOCHS = 25
        return config
    
    @staticmethod
    def ablation_study():
        """Configuration for ablation studies"""
        config = SFTConfig()
        config.N_LAYERS = 1  # Single layer
        config.EPOCHS = 10
        return config

    @staticmethod
    def semantic_composition():
        """Configuration for the semantic composition test"""
        config = SFTConfig()
        config.SEMANTIC_DIM = 64  # Smaller is fine for this conceptual test
        config.FIELD_RESOLUTION = 64
        config.N_LAYERS = 2
        config.EPOCHS = 20  # Needs enough time to learn the concepts
        config.BATCH_SIZE = 4
        config.LEARNING_RATE = 2e-3
        # The number of classes is determined by the dataset (color vs. shape)
        config.N_CLASSES = 2
        return config

    @staticmethod
    def semantic_svo_test():
        """Configuration for the Subject-Verb-Object composition test."""
        config = SFTConfig()
        config.SEMANTIC_DIM = 64
        config.FIELD_RESOLUTION = 64
        config.N_LAYERS = 3  # A bit more depth for the complex relations
        config.EPOCHS = 30  # More epochs to learn the roles
        config.BATCH_SIZE = 4
        config.LEARNING_RATE = 1.5e-3
        # Classification task: Agent (0) vs. Action/Object (1)
        config.N_CLASSES = 2
        return config


# Research-specific configurations
class ResearchConfigs:
    """
    Configurations for specific research directions
    """
    
    @staticmethod
    def locality_study():
        """Study different locality parameters"""
        configs = []
        for width in [0.01, 0.02, 0.05, 0.1]:
            config = SFTConfig()
            config.LOCALITY_WIDTH = width
            configs.append(config)
        return configs
    
    @staticmethod
    def resolution_study():
        """Study different field resolutions"""
        configs = []
        for resolution in [64, 128, 256, 512]:
            config = SFTConfig()
            config.FIELD_RESOLUTION = resolution
            configs.append(config)
        return configs
    
    @staticmethod
    def depth_study():
        """Study different model depths"""
        configs = []
        for n_layers in [1, 2, 3, 4, 5]:
            config = SFTConfig()
            config.N_LAYERS = n_layers
            configs.append(config)
        return configs
