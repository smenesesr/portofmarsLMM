#!/usr/bin/env python3
"""
Batch Experiment Runner for Mars Game

Run multiple experiments with different configurations.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing import load_config, save_results, combine_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_experiment(experiment_config: Dict[str, Any], 
                         base_output_dir: str) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    
    from run_simulation import run_batch_simulation
    
    # Extract experiment parameters
    model = experiment_config['model']
    runs = experiment_config.get('runs', 50)
    risk_level = experiment_config.get('risk_level')
    
    # Create unique output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    risk_suffix = f"_risk{risk_level}" if risk_level is not None else ""
    output_file = f"{base_output_dir}/mars_{model.replace('-', '')}{risk_suffix}_{timestamp}.csv"
    
    logger.info(f"Starting experiment: {model}, {runs} runs, risk_level={risk_level}")
    
    try:
        # Run the simulation
        results_df = run_batch_simulation(
            model_name=model,
            num_runs=runs,
            output_file=output_file,
            config_file=experiment_config.get('config_file'),
            risk_level=risk_level,
            events_file=experiment_config.get('events_file')
        )
        
        # Calculate summary statistics
        summary = {
            'experiment_id': experiment_config.get('id', f"{model}_{risk_level}"),
            'model': model,
            'risk_level': risk_level,
            'runs': len(results_df),
            'success_rate': results_df['failure_round'].isna().mean(),
            'average_score': results_df['final_score'].mean(),
            'median_score': results_df['final_score'].median(),
            'score_std': results_df['final_score'].std(),
            'average_investment': results_df['average_investment'].mean(),
            'output_file': output_file,
            'timestamp': timestamp,
            'status': 'completed'
        }
        
        logger.info(f"Experiment completed: {model}, success_rate={summary['success_rate']:.2%}")
        return summary
        
    except Exception as e:
        logger.error(f"Experiment failed: {model}, error={e}")
        return {
            'experiment_id': experiment_config.get('id', f"{model}_{risk_level}"),
            'model': model,
            'risk_level': risk_level,
            'status': 'failed',
            'error': str(e),
            'timestamp': timestamp
        }


def create_model_comparison_config(models: List[str], runs: int = 50) -> List[Dict[str, Any]]:
    """Create configuration for model comparison experiment."""
    
    experiments = []
    for i, model in enumerate(models):
        experiments.append({
            'id': f'model_comparison_{i+1}',
            'model': model,
            'runs': runs,
            'description': f'Model comparison experiment for {model}'
        })
    
    return experiments


def create_risk_analysis_config(model: str, risk_levels: List[int], 
                               runs: int = 30) -> List[Dict[str, Any]]:
    """Create configuration for risk level analysis."""
    
    experiments = []
    for risk_level in risk_levels:
        experiments.append({
            'id': f'risk_analysis_{risk_level}',
            'model': model,
            'risk_level': risk_level,
            'runs': runs,
            'description': f'Risk analysis for level {risk_level}'
        })
    
    return experiments


def run_batch_experiments(config_file: str, output_dir: str, 
                         max_workers: int = 3) -> pd.DataFrame:
    """Run batch experiments from configuration file."""
    
    # Load experiment configuration
    with open(config_file, 'r') as f:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    experiments = config['experiments']
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting {len(experiments)} experiments with {max_workers} workers")
    
    # Run experiments in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_experiment = {
            executor.submit(run_single_experiment, exp, str(output_dir)): exp
            for exp in experiments
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_experiment):
            experiment = future_to_experiment[future]
            try:
                result = future.result()
                results.append(result)
                
                # Log progress
                completed = len(results)
                total = len(experiments)
                logger.info(f"Progress: {completed}/{total} experiments completed")
                
            except Exception as e:
                logger.error(f"Experiment {experiment.get('id', '?')} failed: {e}")
                results.append({
                    'experiment_id': experiment.get('id', '?'),
                    'status': 'failed',
                    'error': str(e)
                })
    
    # Save experiment summary
    summary_df = pd.DataFrame(results)
    summary_file = output_dir / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    logger.info(f"Batch experiments completed. Summary saved to {summary_file}")
    
    # Combine successful results
    successful_files = [r['output_file'] for r in results if r.get('status') == 'completed']
    if successful_files:
        combined_file = output_dir / f"combined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        combined_df = combine_results(successful_files, str(combined_file))
        logger.info(f"Combined results saved to {combined_file}")
    
    return summary_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run batch Mars game experiments')
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', help='Experiment configuration file (JSON/YAML)')
    group.add_argument('--models', nargs='+', help='Models for comparison experiment')
    group.add_argument('--risk-analysis', help='Model for risk analysis experiment')
    
    # Common options
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--runs', type=int, default=50, help='Number of runs per experiment')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--risk-levels', nargs='+', type=int, 
                       default=[0, 2, 4, 6, 8, 10], help='Risk levels for risk analysis')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        if args.config:
            # Run from configuration file
            summary_df = run_batch_experiments(args.config, args.output_dir, args.workers)
            
        elif args.models:
            # Create and run model comparison
            experiments = create_model_comparison_config(args.models, args.runs)
            config = {'experiments': experiments}
            
            # Save temporary config
            temp_config = Path(args.output_dir) / 'temp_model_comparison.json'
            temp_config.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_config, 'w') as f:
                json.dump(config, f, indent=2)
            
            summary_df = run_batch_experiments(str(temp_config), args.output_dir, args.workers)
            temp_config.unlink()  # Clean up
            
        elif args.risk_analysis:
            # Create and run risk analysis
            experiments = create_risk_analysis_config(args.risk_analysis, args.risk_levels, args.runs)
            config = {'experiments': experiments}
            
            # Save temporary config
            temp_config = Path(args.output_dir) / 'temp_risk_analysis.json'
            temp_config.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_config, 'w') as f:
                json.dump(config, f, indent=2)
            
            summary_df = run_batch_experiments(str(temp_config), args.output_dir, args.workers)
            temp_config.unlink()  # Clean up
        
        # Print final summary
        successful = summary_df['status'] == 'completed'
        print(f"\n{'='*60}")
        print(f"BATCH EXPERIMENTS COMPLETE")
        print(f"{'='*60}")
        print(f"Total Experiments: {len(summary_df)}")
        print(f"Successful: {successful.sum()}")
        print(f"Failed: {(~successful).sum()}")
        
        if successful.any():
            success_df = summary_df[successful]
            print(f"Average Success Rate: {success_df['success_rate'].mean():.2%}")
            print(f"Average Score: {success_df['average_score'].mean():.2f}")
        
        print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Batch experiments interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Batch experiments failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
