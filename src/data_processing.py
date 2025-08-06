"""
Data Processing Utilities for Mars Game

Functions for loading, processing, and saving game data.
"""

import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_results(results: List[Dict[str, Any]], output_path: str, 
                metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save simulation results to CSV with optional metadata."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add metadata as columns if provided
    if metadata:
        for key, value in metadata.items():
            df[f'meta_{key}'] = value
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    # Save metadata separately if provided
    if metadata:
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path}")


def load_risk_profiles(data_path: str, risk_column: str = 'Q12', 
                      skip_rows: int = 2) -> List[int]:
    """
    Load risk tolerance profiles from human behavioral data.
    
    Args:
        data_path: Path to CSV file with human responses
        risk_column: Column name containing risk tolerance scores
        skip_rows: Number of metadata rows to skip
        
    Returns:
        List of risk tolerance scores (0-10)
    """
    df = pd.read_csv(data_path)
    
    # Skip metadata rows and extract risk scores
    risk_scores = df[risk_column].iloc[skip_rows:].astype(int).tolist()
    
    logger.info(f"Loaded {len(risk_scores)} risk profiles from {data_path}")
    return risk_scores


def process_simulation_results(results_path: str) -> Dict[str, Any]:
    """Process and analyze simulation results."""
    
    df = pd.read_csv(results_path)
    
    analysis = {
        'total_runs': len(df),
        'successful_runs': df['failure_round'].isna().sum(),
        'success_rate': df['failure_round'].isna().mean(),
        'average_score': df['final_score'].mean(),
        'median_score': df['final_score'].median(),
        'score_std': df['final_score'].std(),
        'average_survival_rounds': df['survival_rounds'].mean(),
        'average_investment': df['average_investment'].mean(),
        'health_stats': {
            'avg_min_health': df['min_health'].mean(),
            'avg_max_health': df['max_health'].mean(),
            'avg_health_variance': df['health_variance'].mean()
        }
    }
    
    # Add model-specific analysis if multiple models
    if 'model' in df.columns and df['model'].nunique() > 1:
        model_analysis = {}
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            model_analysis[model] = {
                'runs': len(model_df),
                'success_rate': model_df['failure_round'].isna().mean(),
                'average_score': model_df['final_score'].mean(),
                'average_investment': model_df['average_investment'].mean()
            }
        analysis['by_model'] = model_analysis
    
    # Add risk level analysis if available
    if 'risk_level' in df.columns and df['risk_level'].notna().any():
        risk_analysis = {}
        for risk_level in df['risk_level'].dropna().unique():
            risk_df = df[df['risk_level'] == risk_level]
            risk_analysis[int(risk_level)] = {
                'runs': len(risk_df),
                'success_rate': risk_df['failure_round'].isna().mean(),
                'average_score': risk_df['final_score'].mean(),
                'average_investment': risk_df['average_investment'].mean()
            }
        analysis['by_risk_level'] = risk_analysis
    
    return analysis


def combine_results(result_files: List[str], output_path: str) -> pd.DataFrame:
    """Combine multiple result files into a single dataset."""
    
    combined_data = []
    
    for file_path in result_files:
        try:
            df = pd.read_csv(file_path)
            df['source_file'] = Path(file_path).name
            combined_data.append(df)
            logger.info(f"Loaded {len(df)} results from {file_path}")
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    
    if not combined_data:
        raise ValueError("No valid result files found")
    
    # Combine all data
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Save combined results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    logger.info(f"Combined {len(combined_df)} results from {len(result_files)} files")
    logger.info(f"Combined results saved to {output_path}")
    
    return combined_df


def extract_game_trajectories(results_df: pd.DataFrame) -> Dict[str, List[List[int]]]:
    """Extract health and points trajectories from results."""
    
    trajectories = {
        'health': [],
        'points': []
    }
    
    for _, row in results_df.iterrows():
        if pd.notna(row.get('health_trajectory')):
            try:
                # Parse trajectory strings (assuming they're stored as strings)
                health_traj = eval(row['health_trajectory']) if isinstance(row['health_trajectory'], str) else row['health_trajectory']
                points_traj = eval(row['points_trajectory']) if isinstance(row['points_trajectory'], str) else row['points_trajectory']
                
                trajectories['health'].append(health_traj)
                trajectories['points'].append(points_traj)
            except Exception as e:
                logger.warning(f"Failed to parse trajectory for run {row.get('run_id', '?')}: {e}")
    
    return trajectories


def create_summary_report(results_path: str, output_path: str) -> None:
    """Create a summary report from simulation results."""
    
    analysis = process_simulation_results(results_path)
    
    report = f"""
# Mars Game Simulation Report

## Overall Results
- **Total Runs**: {analysis['total_runs']}
- **Successful Runs**: {analysis['successful_runs']} ({analysis['success_rate']:.1%})
- **Average Score**: {analysis['average_score']:.2f} ± {analysis['score_std']:.2f}
- **Median Score**: {analysis['median_score']:.2f}
- **Average Survival**: {analysis['average_survival_rounds']:.1f} rounds
- **Average Investment**: {analysis['average_investment']:.2f} units/round

## Health Management
- **Average Minimum Health**: {analysis['health_stats']['avg_min_health']:.1f}
- **Average Maximum Health**: {analysis['health_stats']['avg_max_health']:.1f}
- **Health Variance**: {analysis['health_stats']['avg_health_variance']:.2f}
"""

    # Add model comparison if available
    if 'by_model' in analysis:
        report += "\n## Model Comparison\n"
        for model, stats in analysis['by_model'].items():
            report += f"- **{model}**: {stats['success_rate']:.1%} success, {stats['average_score']:.1f} avg score\n"
    
    # Add risk level analysis if available
    if 'by_risk_level' in analysis:
        report += "\n## Risk Level Analysis\n"
        for risk_level, stats in analysis['by_risk_level'].items():
            report += f"- **Risk Level {risk_level}**: {stats['success_rate']:.1%} success, {stats['average_score']:.1f} avg score\n"
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Summary report saved to {output_path}")


def validate_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate simulation results for consistency and errors."""
    
    validation = {
        'total_rows': len(results_df),
        'issues': []
    }
    
    # Check for missing critical columns
    required_columns = ['final_score', 'survival_rounds', 'total_investment']
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    if missing_columns:
        validation['issues'].append(f"Missing columns: {missing_columns}")
    
    # Check for invalid scores
    if 'final_score' in results_df.columns:
        negative_scores = (results_df['final_score'] < 0).sum()
        if negative_scores > 0:
            validation['issues'].append(f"{negative_scores} runs with negative scores")
    
    # Check for invalid survival rounds
    if 'survival_rounds' in results_df.columns:
        invalid_rounds = ((results_df['survival_rounds'] < 1) | (results_df['survival_rounds'] > 11)).sum()
        if invalid_rounds > 0:
            validation['issues'].append(f"{invalid_rounds} runs with invalid survival rounds")
    
    # Check for consistency between failure_round and survival_rounds
    if all(col in results_df.columns for col in ['failure_round', 'survival_rounds']):
        inconsistent = 0
        for _, row in results_df.iterrows():
            if pd.notna(row['failure_round']) and row['failure_round'] != row['survival_rounds']:
                inconsistent += 1
        if inconsistent > 0:
            validation['issues'].append(f"{inconsistent} runs with inconsistent failure/survival rounds")
    
    validation['is_valid'] = len(validation['issues']) == 0
    
    return validation
