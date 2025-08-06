#!/usr/bin/env python3
"""
Mars Game Simulation Runner

Main script for running Mars resource management game simulations.
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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from game_engine import MarsGame, GameResult
from ai_agents import create_agent, AgentConfig
from data_processing import save_results, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_simulation(game: MarsGame, agent, run_id: int) -> Dict[str, Any]:
    """Run a single game simulation."""
    
    # Reset game state
    game.event_deck.reset()
    state = game.create_initial_state()
    
    # Track game progress
    health_trajectory = [state.health]
    points_trajectory = [state.total_points]
    all_events = []
    total_investment = 0
    
    logger.debug(f"Starting simulation {run_id}")
    
    # Game loop
    while not game.is_game_over(state):
        # Agent makes decision
        try:
            investment = agent.make_decision(state, {
                'round': state.round_number,
                'max_rounds': game.max_rounds
            })
            total_investment += investment
            
            # Simulate round
            state = game.simulate_round(state, investment)
            
            # Track progress
            health_trajectory.append(state.health)
            points_trajectory.append(state.total_points)
            all_events.extend(state.events_this_round)
            
            # Move to next round
            state.round_number += 1
            state.events_this_round = []
            state.investment_capacity = 10  # Reset capacity
            
        except Exception as e:
            logger.error(f"Error in simulation {run_id}, round {state.round_number}: {e}")
            state.game_over = True
            state.failure_round = state.round_number
    
    # Create result
    result = game.create_result(
        state, health_trajectory, points_trajectory, 
        all_events, total_investment
    )
    
    logger.debug(f"Simulation {run_id} completed: Score={result.final_score}, Rounds={result.survival_rounds}")
    
    return {
        'run_id': run_id,
        'final_score': result.final_score,
        'survival_rounds': result.survival_rounds,
        'failure_round': result.failure_round,
        'total_investment': result.total_investment,
        'average_investment': result.average_investment,
        'min_health': result.strategy_summary['min_health'],
        'max_health': result.strategy_summary['max_health'],
        'avg_health': result.strategy_summary['avg_health'],
        'health_variance': result.strategy_summary['health_variance'],
        'events_count': len(result.events_encountered),
        'health_trajectory': result.health_trajectory,
        'points_trajectory': result.points_trajectory,
        'timestamp': datetime.now().isoformat()
    }


def run_batch_simulation(
    model_name: str,
    num_runs: int,
    output_file: str,
    config_file: Optional[str] = None,
    risk_level: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """Run multiple simulations and save results."""
    
    logger.info(f"Starting batch simulation: {model_name}, {num_runs} runs")
    
    # Load configuration
    if config_file and Path(config_file).exists():
        config = load_config(config_file)
    else:
        config = {
            'game': {
                'starting_health': 15,
                'max_health': 25,
                'max_rounds': 11
            },
            'simulation': {
                'temperature': 0.7,
                'delay_between_calls': 1.0
            }
        }
    
    # Initialize game
    game = MarsGame(
        starting_health=config['game']['starting_health'],
        max_health=config['game']['max_health'],
        max_rounds=config['game']['max_rounds'],
        events_file=kwargs.get('events_file')
    )
    
    # Create agent
    agent_config = AgentConfig(
        model_name=model_name,
        temperature=config['simulation']['temperature'],
        api_key=os.getenv('OPENAI_API_KEY'),
        delay_between_calls=config['simulation']['delay_between_calls']
    )
    
    if risk_level is not None:
        agent = create_agent('risk_aware', agent_config, risk_level=risk_level)
    else:
        agent = create_agent('gpt', agent_config)
    
    # Run simulations
    results = []
    for i in range(num_runs):
        try:
            result = run_single_simulation(game, agent, i + 1)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{num_runs} simulations")
                
        except Exception as e:
            logger.error(f"Failed simulation {i + 1}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add metadata
    df['model'] = model_name
    df['config_file'] = config_file
    df['risk_level'] = risk_level
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Summary: {len(results)} successful runs, avg score: {df['final_score'].mean():.2f}")
    
    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Mars game simulations')
    
    parser.add_argument('--model', required=True, 
                       help='AI model to use (e.g., gpt-4, gpt-3.5-turbo)')
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of simulation runs')
    parser.add_argument('--output', required=True,
                       help='Output CSV file path')
    parser.add_argument('--config', 
                       help='Configuration file path')
    parser.add_argument('--events-file',
                       help='Events CSV file path')
    parser.add_argument('--risk-level', type=int,
                       help='Risk level for risk-aware agent (0-10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.risk_level is not None and not (0 <= args.risk_level <= 10):
        parser.error("Risk level must be between 0 and 10")
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        # Run simulation
        results_df = run_batch_simulation(
            model_name=args.model,
            num_runs=args.runs,
            output_file=args.output,
            config_file=args.config,
            risk_level=args.risk_level,
            events_file=args.events_file
        )
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*50}")
        print(f"Model: {args.model}")
        print(f"Runs: {len(results_df)}")
        print(f"Average Score: {results_df['final_score'].mean():.2f}")
        print(f"Success Rate: {(results_df['failure_round'].isna().sum() / len(results_df) * 100):.1f}%")
        print(f"Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
