# #!/usr/bin/env python3
# """
# Simple script to run SFT experiments
# Easy entry point for research experiments
# """

# from experiments.experiment_runner import SFTExperiment, quick_experiment
# from config.config import SFTConfig, ExperimentConfigs, ResearchConfigs


# def run_quick_test():
#     """Run a quick test experiment"""
#     print("🚀 Running quick test experiment...")
#     return quick_experiment(semantic_dim=64, field_resolution=64, epochs=5)


# def run_full_experiment():
#     """Run the full SFT experiment"""
#     print("🚀 Running full SFT experiment...")
    
#     experiment = SFTExperiment(
#         semantic_dim=128,
#         field_resolution=128,
#         n_layers=3,
#         n_classes=2
#     )
    
#     return experiment.run_complete_experiment(
#         epochs=15,
#         lr=1e-3,
#         batch_size=8
#     )


# def run_custom_experiment(config_name='full'):
#     """Run experiment with predefined configuration"""
    
#     if config_name == 'quick':
#         config = ExperimentConfigs.quick_test()
#     elif config_name == 'full':
#         config = ExperimentConfigs.full_experiment()
#     elif config_name == 'high_res':
#         config = ExperimentConfigs.high_resolution()
#     elif config_name == 'ablation':
#         config = ExperimentConfigs.ablation_study()
#     else:
#         config = SFTConfig()
    
#     experiment = SFTExperiment(
#         semantic_dim=config.SEMANTIC_DIM,
#         field_resolution=config.FIELD_RESOLUTION,
#         n_layers=config.N_LAYERS,
#         n_classes=config.N_CLASSES,
#         device=config.DEVICE
#     )
    
#     experiment.setup_data(
#         batch_size=config.BATCH_SIZE,
#         test_split=config.TEST_SPLIT
#     )
    
#     training_history = experiment.train(
#         epochs=config.EPOCHS,
#         lr=config.LEARNING_RATE
#     )
    
#     test_results = experiment.evaluate()
#     validation_results = experiment.run_validations()
#     experiment.create_visualizations()
    
#     return {
#         'experiment': experiment,
#         'training_history': training_history,
#         'test_results': test_results,
#         'validation_results': validation_results
#     }


# def run_research_study(study_type='locality'):
#     """Run research studies with multiple configurations"""
    
#     if study_type == 'locality':
#         configs = ResearchConfigs.locality_study()
#     elif study_type == 'resolution':
#         configs = ResearchConfigs.resolution_study()
#     elif study_type == 'depth':
#         configs = ResearchConfigs.depth_study()
#     else:
#         raise ValueError(f"Unknown study type: {study_type}")
    
#     results = []
    
#     for i, config in enumerate(configs):
#         print(f"\n{'='*60}")
#         print(f"Running {study_type} study {i+1}/{len(configs)}")
#         print(f"{'='*60}")
        
#         experiment = SFTExperiment(
#             semantic_dim=config.SEMANTIC_DIM,
#             field_resolution=config.FIELD_RESOLUTION,
#             n_layers=config.N_LAYERS,
#             n_classes=config.N_CLASSES
#         )
        
#         experiment.setup_data(batch_size=config.BATCH_SIZE)
#         training_history = experiment.train(epochs=config.EPOCHS, lr=config.LEARNING_RATE)
#         test_results = experiment.evaluate()
        
#         results.append({
#             'config': config,
#             'training_history': training_history,
#             'test_results': test_results,
#             'final_accuracy': test_results['accuracy'] if test_results else 0
#         })
    
#     # Print study summary
#     print(f"\n{'='*80}")
#     print(f"{study_type.upper()} STUDY SUMMARY")
#     print(f"{'='*80}")
    
#     for i, result in enumerate(results):
#         config = result['config']
#         acc = result['final_accuracy']
        
#         if study_type == 'locality':
#             param = config.LOCALITY_WIDTH
#             print(f"Locality Width {param:.3f}: Accuracy = {acc:.4f}")
#         elif study_type == 'resolution':
#             param = config.FIELD_RESOLUTION
#             print(f"Resolution {param:3d}: Accuracy = {acc:.4f}")
#         elif study_type == 'depth':
#             param = config.N_LAYERS
#             print(f"Layers {param:1d}: Accuracy = {acc:.4f}")
    
#     return results


# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Run SFT experiments')
#     parser.add_argument('--mode', choices=['quick', 'full', 'custom', 'study'], 
#                        default='full', help='Experiment mode')
#     parser.add_argument('--config', choices=['quick', 'full', 'high_res', 'ablation'], 
#                        default='full', help='Configuration preset')
#     parser.add_argument('--study', choices=['locality', 'resolution', 'depth'], 
#                        default='locality', help='Research study type')
    
#     args = parser.parse_args()
    
#     if args.mode == 'quick':
#         results = run_quick_test()
#     elif args.mode == 'full':
#         results = run_full_experiment()
#     elif args.mode == 'custom':
#         results = run_custom_experiment(args.config)
#     elif args.mode == 'study':
#         results = run_research_study(args.study)
    
#     print("\n✅ Experiment completed successfully!")

#!/usr/bin/env python3
"""
Entry point for running SFT experiments.
"""
import argparse
from config.config import ExperimentConfigs
from experiments.experiment_runner import SFTExperiment

def main():
    parser = argparse.ArgumentParser(description="Run Semantic Field Transform Experiments")
    parser.add_argument(
        '--experiment', 
        type=str, 
        default='svo', 
        choices=['quick', 'full', 'composition', 'svo'],
        help='The type of experiment to run.'
    )
    args = parser.parse_args()

    if args.experiment == 'composition':
        config = ExperimentConfigs.semantic_composition()
        experiment = SFTExperiment(
            semantic_dim=config.SEMANTIC_DIM, field_resolution=config.FIELD_RESOLUTION,
            n_layers=config.N_LAYERS, n_classes=config.N_CLASSES
        )
        experiment.run_composition_experiment(epochs=config.EPOCHS, lr=config.LEARNING_RATE)

    elif args.experiment == 'svo':
        config = ExperimentConfigs.semantic_svo_test()
        experiment = SFTExperiment(
            semantic_dim=config.SEMANTIC_DIM, field_resolution=config.FIELD_RESOLUTION,
            n_layers=config.N_LAYERS, n_classes=config.N_CLASSES
        )
        experiment.run_svo_experiment(epochs=config.EPOCHS, lr=config.LEARNING_RATE)

    elif args.experiment == 'quick':
        config = ExperimentConfigs.quick_test()
        experiment = SFTExperiment(
            semantic_dim=config.SEMANTIC_DIM, field_resolution=config.FIELD_RESOLUTION,
            n_layers=config.N_LAYERS
        )
        experiment.run_complete_experiment(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
        
    elif args.experiment == 'full':
        config = ExperimentConfigs.full_experiment()
        experiment = SFTExperiment(
            semantic_dim=config.SEMANTIC_DIM, field_resolution=config.FIELD_RESOLUTION,
            n_layers=config.N_LAYERS
        )
        experiment.run_complete_experiment(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

if __name__ == "__main__":
    main()