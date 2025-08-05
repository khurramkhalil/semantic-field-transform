#!/usr/bin/env python3
"""
SFT v2 Relational Prediction Experiment Runner
Orchestrates the MLM training and subsequent SVO validation.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.sft_model import CompleteGenuineSFT_v2
from utils.data_utils import create_mlm_dataset, SFT_MLM_Dataset
from utils.training_utils import SFT_MLM_Trainer
from utils.validation_utils import run_svo_composition_test
from utils.visualization_utils import plot_similarity_heatmap
from torch.utils.data import DataLoader

from utils.training_utils import SFT_Contrastive_Trainer # Import the new trainer

class SFT_Relational_Experiment:
    """
    The definitive experiment runner, now using a robust Contrastive trainer.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        print(f"Using device: {self.device}")
        
        self.model = CompleteGenuineSFT_v2(
            semantic_dim=config.SEMANTIC_DIM,
            field_resolution=config.FIELD_RESOLUTION,
            n_layers=config.N_LAYERS,
            n_components=2,
            n_classes=config.N_CLASSES
        ).to(self.device)
        
    def run(self):
        """
        Run the complete relational prediction experiment pipeline with contrastive learning.
        """
        print("\n🔬 RUNNING RELATIONAL PREDICTION EXPERIMENT (CONTRASTIVE)")
        print("=" * 80)
        
        # 1. Create the specialized MLM dataset
        training_pairs = create_mlm_dataset()
        mlm_dataset = SFT_MLM_Dataset(training_pairs, max_length=self.config.MAX_LENGTH)
        train_loader = DataLoader(mlm_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        # 2. Train the model using the new CONTRASTIVE trainer
        print("\n🚀 TRAINING SFT v2 WITH CONTRASTIVE (INFONCE) LOSS")
        trainer = SFT_Contrastive_Trainer(self.model, device=self.device, lr=self.config.LEARNING_RATE)
        trainer.train(train_loader, epochs=self.config.EPOCHS, verbose=True)
        print("Training complete.")

        # 3. After training, run the SVO validation to see if it learned roles
        from utils.data_utils import create_svo_compositional_dataset
        svo_validation_dataset = create_svo_compositional_dataset()
        
        similarity_matrix, labels = run_svo_composition_test(
            self.model, svo_validation_dataset, device=self.device
        )
        
        # 4. Visualize the results
        plot_similarity_heatmap(
            similarity_matrix, 
            labels, 
            title="SFT v2 (Contrastive Trained): SVO Semantic Role Similarity",
            save_path="svo_composition_v2_contrastive.png"
        )
        
        print("\n✅ Contrastive Relational Experiment Finished.")