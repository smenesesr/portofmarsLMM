"""
Mars Resource Management Game Engine

Core game mechanics and simulation logic for the Mars colonization resource game.
"""

import random
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Represents the current state of the game."""
    round_number: int
    health: int
    total_points: int
    investment_capacity: int = 10
    events_this_round: List[Dict] = None
    game_over: bool = False
    failure_round: Optional[int] = None

    def __post_init__(self):
        if self.events_this_round is None:
            self.events_this_round = []


@dataclass
class GameResult:
    """Final result of a game simulation."""
    final_score: int
    survival_rounds: int
    failure_round: Optional[int]
    total_investment: int
    average_investment: float
    health_trajectory: List[int]
    points_trajectory: List[int]
    events_encountered: List[Dict]
    strategy_summary: Dict[str, Any]


class EventDeck:
    """Manages the game events and their effects."""
    
    def __init__(self, events_file: Optional[str] = None):
        """Initialize event deck from CSV file or use default events."""
        if events_file and Path(events_file).exists():
            self.events = self._load_events_from_csv(events_file)
        else:
            self.events = self._get_default_events()
        
        self.current_event_index = 0
        logger.info(f"Initialized event deck with {len(self.events)} events")

    def _load_events_from_csv(self, filepath: str) -> List[Dict]:
        """Load events from CSV file."""
        df = pd.read_csv(filepath)
        return df.to_dict(orient='records')

    def _get_default_events(self) -> List[Dict]:
        """Default event set if no CSV provided."""
        return [
            {"id": 1, "type": "health", "value": 7, "description": "Increase system health with 7 units"},
            {"id": 2, "type": "health", "value": -6, "description": "Reduce system health with 6 units"},
            {"id": 3, "type": "none", "value": 0, "description": "Nothing happens"},
            {"id": 4, "type": "points", "value": 5, "description": "Increase points gained with 5 points"},
            {"id": 5, "type": "health", "value": -5, "description": "Reduce system health with 5 units"},
            {"id": 6, "type": "points", "value": 4, "description": "Increase points gained with 4 points"},
            {"id": 7, "type": "points", "value": 2, "description": "Increase points gained with 2 points"},
            {"id": 8, "type": "points", "value": -4, "description": "Reduce points gained with 4 points"},
            {"id": 9, "type": "none", "value": 0, "description": "Nothing happens"},
            {"id": 10, "type": "none", "value": 0, "description": "Nothing happens"},
            {"id": 11, "type": "extra_events", "value": 2, "description": "2 additional events will happen this round"},
            {"id": 12, "type": "cap", "value": -6, "description": "Reduce investment capacity by 6 units"},
            {"id": 13, "type": "none", "value": 0, "description": "Nothing happens"},
            {"id": 14, "type": "min_invest", "value": 8, "description": "Must invest at least 8 units"},
            {"id": 15, "type": "health", "value": -1, "description": "Reduce system health with 1 unit"},
            {"id": 16, "type": "points", "value": -1, "description": "Reduce points gained with 1 point"},
            {"id": 17, "type": "points", "value": 9, "description": "Increase points gained with 9 points"},
            {"id": 18, "type": "health", "value": -8, "description": "Reduce system health with 8 units"},
            {"id": 19, "type": "none", "value": 0, "description": "Nothing happens"},
            {"id": 20, "type": "none", "value": 0, "description": "Nothing happens"},
            {"id": 21, "type": "none", "value": 0, "description": "Nothing happens"},
            {"id": 22, "type": "health", "value": 3, "description": "Increase system health with 3 units"},
            {"id": 23, "type": "points", "value": -5, "description": "Reduce points gained with 5 points"},
            {"id": 24, "type": "health", "value": 6, "description": "Increase system health with 6 units"},
            {"id": 25, "type": "none", "value": 0, "description": "Nothing happens"},
            {"id": 26, "type": "cap", "value": 2, "description": "Increase investment capacity by 2 points"},
            {"id": 27, "type": "health", "value": 4, "description": "Increase system health with 4 units"},
            {"id": 28, "type": "none", "value": 0, "description": "Nothing happens"},
            {"id": 29, "type": "points", "value": -10, "description": "Reduce points gained with 10 points"},
            {"id": 30, "type": "none", "value": 0, "description": "Nothing happens"},
        ]

    def get_next_events(self, num_events: int) -> List[Dict]:
        """Get the next N events from the deck."""
        events = []
        for _ in range(num_events):
            if self.current_event_index < len(self.events):
                events.append(self.events[self.current_event_index])
                self.current_event_index += 1
        return events

    def reset(self):
        """Reset event deck to beginning."""
        self.current_event_index = 0


class MarsGame:
    """Main game engine for Mars resource management simulation."""
    
    def __init__(self, 
                 starting_health: int = 15,
                 max_health: int = 25,
                 max_rounds: int = 11,
                 events_file: Optional[str] = None):
        """
        Initialize the Mars game.
        
        Args:
            starting_health: Initial system health
            max_health: Maximum possible health
            max_rounds: Total number of rounds
            events_file: Path to events CSV file
        """
        self.starting_health = starting_health
        self.max_health = max_health
        self.max_rounds = max_rounds
        self.event_deck = EventDeck(events_file)
        
        logger.info(f"Initialized Mars game: {max_rounds} rounds, health {starting_health}-{max_health}")

    def _calculate_num_events(self, health: int) -> int:
        """Calculate number of events based on current health."""
        if health >= 16:
            return 1
        elif health >= 9:
            return 2
        else:
            return 3

    def _apply_investment(self, state: GameState, investment: int) -> Tuple[int, int]:
        """
        Apply investment decision and return (new_health, points_earned).
        
        Args:
            state: Current game state
            investment: Amount to invest in health
            
        Returns:
            Tuple of (new_health, points_earned)
        """
        # Validate investment
        max_investment = min(state.investment_capacity, 10)
        investment = max(0, min(investment, max_investment))
        
        # Calculate new health (capped at max_health)
        new_health = min(state.health + investment, self.max_health)
        
        # Calculate points earned (remaining capacity)
        points_earned = state.investment_capacity - investment
        
        return new_health, points_earned

    def _apply_events(self, state: GameState, events: List[Dict]) -> Tuple[int, int]:
        """
        Apply events to current state.
        
        Returns:
            Tuple of (health_change, points_change)
        """
        total_health_change = 0
        total_points_change = 0
        extra_events = 0
        
        for event in events:
            event_type = event.get("type", "none")
            value = event.get("value", 0)
            
            if event_type == "health":
                total_health_change += value
            elif event_type == "points":
                total_points_change += value
            elif event_type == "extra_events":
                extra_events += value
            elif event_type == "cap":
                state.investment_capacity = max(0, state.investment_capacity + value)
            elif event_type == "min_invest":
                # This would need to be handled by the AI agent
                pass
        
        # Handle extra events
        if extra_events > 0:
            additional_events = self.event_deck.get_next_events(extra_events)
            state.events_this_round.extend(additional_events)
            # Recursively apply additional events
            add_health, add_points = self._apply_events(state, additional_events)
            total_health_change += add_health
            total_points_change += add_points
        
        return total_health_change, total_points_change

    def simulate_round(self, state: GameState, investment_decision: int) -> GameState:
        """
        Simulate a single round of the game.
        
        Args:
            state: Current game state
            investment_decision: AI's investment decision
            
        Returns:
            Updated game state
        """
        # 1. Automatic health decay
        state.health -= 5
        
        # 2. Check for immediate failure
        if state.health <= 0:
            state.game_over = True
            state.failure_round = state.round_number
            return state
        
        # 3. Determine and apply events
        num_events = self._calculate_num_events(state.health)
        events = self.event_deck.get_next_events(num_events)
        state.events_this_round = events.copy()
        
        health_change, points_change = self._apply_events(state, events)
        state.health = max(0, min(state.health + health_change, self.max_health))
        
        # 4. Check for failure after events
        if state.health <= 0:
            state.game_over = True
            state.failure_round = state.round_number
            return state
        
        # 5. Apply investment decision
        new_health, points_earned = self._apply_investment(state, investment_decision)
        state.health = new_health
        state.total_points += points_earned + points_change
        
        # 6. Final health check
        if state.health <= 0:
            state.game_over = True
            state.failure_round = state.round_number
        
        return state

    def create_initial_state(self) -> GameState:
        """Create initial game state."""
        return GameState(
            round_number=1,
            health=self.starting_health,
            total_points=0,
            investment_capacity=10
        )

    def is_game_over(self, state: GameState) -> bool:
        """Check if game should end."""
        return (state.game_over or 
                state.round_number > self.max_rounds or 
                state.health <= 0)

    def create_result(self, state: GameState, 
                     health_trajectory: List[int],
                     points_trajectory: List[int],
                     all_events: List[Dict],
                     total_investment: int) -> GameResult:
        """Create final game result."""
        
        final_score = 0 if state.failure_round else state.total_points
        survival_rounds = state.failure_round if state.failure_round else self.max_rounds
        
        return GameResult(
            final_score=final_score,
            survival_rounds=survival_rounds,
            failure_round=state.failure_round,
            total_investment=total_investment,
            average_investment=total_investment / max(1, survival_rounds),
            health_trajectory=health_trajectory,
            points_trajectory=points_trajectory,
            events_encountered=all_events,
            strategy_summary={
                "avg_health": sum(health_trajectory) / len(health_trajectory),
                "min_health": min(health_trajectory),
                "max_health": max(health_trajectory),
                "health_variance": pd.Series(health_trajectory).var()
            }
        )
