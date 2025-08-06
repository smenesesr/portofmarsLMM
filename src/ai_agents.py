"""
AI Agents for Mars Resource Management Game

Different AI agent implementations for playing the Mars game.
"""

import openai
import time
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path

from .game_engine import GameState

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for AI agents."""
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    api_key: Optional[str] = None
    delay_between_calls: float = 1.0
    max_retries: int = 3


class BaseAgent(ABC):
    """Abstract base class for all AI agents."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.call_count = 0
        
    @abstractmethod
    def make_decision(self, state: GameState, game_context: Dict[str, Any]) -> int:
        """Make investment decision based on current game state."""
        pass
    
    def _add_delay(self):
        """Add delay between API calls to respect rate limits."""
        if self.config.delay_between_calls > 0:
            time.sleep(self.config.delay_between_calls)


class GPTAgent(BaseAgent):
    """OpenAI GPT-based agent for playing Mars game."""
    
    def __init__(self, config: AgentConfig, prompt_template: Optional[str] = None):
        super().__init__(config)
        self.client = openai.OpenAI(api_key=config.api_key)
        self.prompt_template = prompt_template or self._get_default_prompt()
        
    def _get_default_prompt(self) -> str:
        """Get default game prompt."""
        return """
You are playing a resource management game set on Mars. Here are the rules:

GAME RULES:
- You start with 15 system health units
- Each round, system health automatically decreases by 5 units
- You can invest 0-10 units in system health each round
- Units not invested become points you earn
- Maximum system health is 25 units
- If system health reaches 0, your total score becomes 0
- The game lasts 11 rounds

EVENTS:
Events happen each round based on your current health:
- Health 16-25: 1 event per round
- Health 9-15: 2 events per round  
- Health <9: 3 events per round

CURRENT SITUATION:
Round: {round_number}
Current Health: {health}
Total Points So Far: {total_points}
Investment Capacity: {investment_capacity}

EVENTS THIS ROUND:
{events_description}

Health after automatic decay (-5): {health_after_decay}
Health after events: {health_after_events}

DECISION REQUIRED:
How many units (0-{max_investment}) do you want to invest in system health?
Units not invested will be added to your points.

Respond with only a number between 0 and {max_investment}.
"""

    def _format_events(self, events: List[Dict]) -> str:
        """Format events for prompt."""
        if not events:
            return "No events this round."
        
        event_descriptions = []
        for i, event in enumerate(events, 1):
            desc = event.get('description', f"Event {event.get('id', '?')}")
            event_descriptions.append(f"Event {i}: {desc}")
        
        return "\n".join(event_descriptions)

    def _extract_investment_decision(self, response: str) -> int:
        """Extract investment decision from GPT response."""
        # Look for numbers in the response
        numbers = re.findall(r'\b(\d+)\b', response.strip())
        
        if numbers:
            # Take the first number found
            decision = int(numbers[0])
            return max(0, min(decision, 10))  # Clamp to valid range
        
        # Default to conservative investment if no clear number found
        logger.warning(f"Could not extract clear decision from: {response[:100]}...")
        return 5

    def make_decision(self, state: GameState, game_context: Dict[str, Any]) -> int:
        """Make investment decision using GPT."""
        
        # Calculate health after decay and events for context
        health_after_decay = max(0, state.health - 5)
        
        # Format events description
        events_desc = self._format_events(state.events_this_round)
        
        # Prepare prompt
        prompt = self.prompt_template.format(
            round_number=state.round_number,
            health=state.health,
            total_points=state.total_points,
            investment_capacity=state.investment_capacity,
            events_description=events_desc,
            health_after_decay=health_after_decay,
            health_after_events=state.health,  # This would be calculated in game engine
            max_investment=min(state.investment_capacity, 10)
        )
        
        # Make API call with retries
        for attempt in range(self.config.max_retries):
            try:
                self._add_delay()
                
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                decision_text = response.choices[0].message.content.strip()
                decision = self._extract_investment_decision(decision_text)
                
                self.call_count += 1
                logger.debug(f"GPT decision: {decision} (from: {decision_text[:50]}...)")
                
                return decision
                
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error("All API call attempts failed, using default decision")
                    return 5  # Conservative default
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return 5  # Fallback


class RiskAwareAgent(GPTAgent):
    """GPT agent that incorporates risk tolerance from human behavioral data."""
    
    def __init__(self, config: AgentConfig, risk_level: int, prompt_template: Optional[str] = None):
        """
        Initialize risk-aware agent.
        
        Args:
            config: Agent configuration
            risk_level: Risk tolerance level (0-10, where 0 is risk-averse, 10 is risk-seeking)
            prompt_template: Custom prompt template
        """
        self.risk_level = risk_level
        super().__init__(config, prompt_template or self._get_risk_aware_prompt())
    
    def _get_risk_aware_prompt(self) -> str:
        """Get risk-aware prompt template."""
        risk_description = self._get_risk_description(self.risk_level)
        
        base_prompt = self._get_default_prompt()
        risk_context = f"""
PERSONALITY CONTEXT:
You recently were asked how you see yourself, if you are generally a person who is fully prepared to take risks or do you try to avoid risks. You were asked to respond on a scale from 0 to 10, where 0 means 'I'm not at all willing to take risks' and 10 means 'I'm very willing to take risks'. You answered: {self.risk_level}

This means you are: {risk_description}

Consider this personality trait when making your investment decisions.

"""
        
        return risk_context + base_prompt
    
    def _get_risk_description(self, risk_level: int) -> str:
        """Get description of risk level."""
        if risk_level <= 2:
            return "very risk-averse and prefer safe, conservative strategies"
        elif risk_level <= 4:
            return "somewhat risk-averse and prefer cautious approaches"
        elif risk_level <= 6:
            return "moderately balanced between risk and safety"
        elif risk_level <= 8:
            return "somewhat risk-seeking and willing to take calculated risks"
        else:
            return "very risk-seeking and comfortable with high-risk strategies"


class HumanAgent(BaseAgent):
    """Human player agent for interactive testing."""
    
    def make_decision(self, state: GameState, game_context: Dict[str, Any]) -> int:
        """Get investment decision from human player."""
        print(f"\n--- Round {state.round_number} ---")
        print(f"Current Health: {state.health}")
        print(f"Total Points: {state.total_points}")
        print(f"Investment Capacity: {state.investment_capacity}")
        
        if state.events_this_round:
            print("Events this round:")
            for i, event in enumerate(state.events_this_round, 1):
                print(f"  {i}. {event.get('description', 'Unknown event')}")
        
        while True:
            try:
                max_investment = min(state.investment_capacity, 10)
                decision = int(input(f"Investment (0-{max_investment}): "))
                if 0 <= decision <= max_investment:
                    return decision
                else:
                    print(f"Please enter a number between 0 and {max_investment}")
            except ValueError:
                print("Please enter a valid number")


class RandomAgent(BaseAgent):
    """Random agent for baseline comparison."""
    
    def __init__(self, config: AgentConfig, seed: Optional[int] = None):
        super().__init__(config)
        if seed is not None:
            import random
            random.seed(seed)
    
    def make_decision(self, state: GameState, game_context: Dict[str, Any]) -> int:
        """Make random investment decision."""
        import random
        max_investment = min(state.investment_capacity, 10)
        return random.randint(0, max_investment)


def create_agent(agent_type: str, config: AgentConfig, **kwargs) -> BaseAgent:
    """Factory function to create agents."""
    
    if agent_type.lower() == "gpt":
        return GPTAgent(config, **kwargs)
    elif agent_type.lower() == "risk_aware":
        return RiskAwareAgent(config, **kwargs)
    elif agent_type.lower() == "human":
        return HumanAgent(config)
    elif agent_type.lower() == "random":
        return RandomAgent(config, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
