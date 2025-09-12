"""
Explanation Logger for Agent Explainability
Provides logging and audit trail functionality for agent decisions and actions.
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from models import (
    AgentExplanation, ChainOfThought, ReActTrace, 
    ThoughtStep, ActionStep, ObservationStep
)


class ExplanationLogger:
    """Logger for agent explanations and audit trails"""
    
    def __init__(self, log_dir: str = "explanation_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.explanations: List[AgentExplanation] = []
    
    def log_explanation(self, explanation: AgentExplanation) -> None:
        """Log an agent explanation"""
        self.explanations.append(explanation)
        
        # Save to individual file
        filename = f"{self.session_id}_{explanation.agent_name}_{explanation.timestamp.strftime('%H%M%S')}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(explanation.model_dump(), f, indent=2, default=str)
    
    def get_explanations_for_agent(self, agent_name: str) -> List[AgentExplanation]:
        """Get all explanations for a specific agent"""
        return [exp for exp in self.explanations if exp.agent_name == agent_name]
    
    def get_session_explanations(self) -> List[AgentExplanation]:
        """Get all explanations for the current session"""
        return self.explanations.copy()
    
    def save_session_summary(self) -> str:
        """Save a summary of all explanations in the current session"""
        summary = {
            "session_id": self.session_id,
            "total_explanations": len(self.explanations),
            "agents": list(set(exp.agent_name for exp in self.explanations)),
            "explanations": [exp.model_dump() for exp in self.explanations]
        }
        
        filename = f"{self.session_id}_session_summary.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return str(filepath)
    
    def create_audit_trail(self) -> Dict[str, Any]:
        """Create a comprehensive audit trail of all agent decisions"""
        audit_trail = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "total_agents": len(set(exp.agent_name for exp in self.explanations)),
            "total_decisions": len(self.explanations),
            "agent_summaries": {}
        }
        
        # Group explanations by agent
        agent_groups = {}
        for exp in self.explanations:
            if exp.agent_name not in agent_groups:
                agent_groups[exp.agent_name] = []
            agent_groups[exp.agent_name].append(exp)
        
        # Create summaries for each agent
        for agent_name, explanations in agent_groups.items():
            avg_confidence = sum(exp.confidence_score for exp in explanations) / len(explanations)
            avg_processing_time = sum(exp.processing_time for exp in explanations) / len(explanations)
            
            audit_trail["agent_summaries"][agent_name] = {
                "total_decisions": len(explanations),
                "average_confidence": round(avg_confidence, 3),
                "average_processing_time": round(avg_processing_time, 3),
                "decision_factors": list(set(factor for exp in explanations for factor in exp.decision_factors)),
                "limitations": list(set(limitation for exp in explanations for limitation in exp.limitations))
            }
        
        return audit_trail


class ExplanationBuilder:
    """Builder for creating agent explanations with Chain-of-Thought and ReAct patterns"""
    
    def __init__(self, agent_name: str, task_description: str):
        self.agent_name = agent_name
        self.task_description = task_description
        self.start_time = time.time()
        self.thought_steps: List[ThoughtStep] = []
        self.action_steps: List[ActionStep] = []
        self.observation_steps: List[ObservationStep] = []
        self.decision_factors: List[str] = []
        self.alternative_approaches: List[str] = []
        self.limitations: List[str] = []
        self.input_summary: Dict[str, Any] = {}
        self.output_summary: Dict[str, Any] = {}
    
    def add_thought(self, thought: str, confidence: float = 0.8) -> 'ExplanationBuilder':
        """Add a thought step to the Chain-of-Thought reasoning"""
        step = ThoughtStep(
            step_number=len(self.thought_steps) + 1,
            thought=thought,
            confidence=confidence
        )
        self.thought_steps.append(step)
        return self
    
    def add_action(self, action_type: str, action_description: str, 
                   input_data: Dict[str, Any], output_data: Dict[str, Any], 
                   success: bool = True, error_message: Optional[str] = None) -> 'ExplanationBuilder':
        """Add an action step to the ReAct trace"""
        step = ActionStep(
            action_type=action_type,
            action_description=action_description,
            input_data=input_data,
            output_data=output_data,
            success=success,
            error_message=error_message
        )
        self.action_steps.append(step)
        return self
    
    def add_observation(self, observation_type: str, observation_content: str, 
                       source: str = "llm", relevance_score: float = 0.8) -> 'ExplanationBuilder':
        """Add an observation step to the ReAct trace"""
        step = ObservationStep(
            observation_type=observation_type,
            observation_content=observation_content,
            source=source,
            relevance_score=relevance_score
        )
        self.observation_steps.append(step)
        return self
    
    def add_decision_factor(self, factor: str) -> 'ExplanationBuilder':
        """Add a factor that influenced the decision"""
        self.decision_factors.append(factor)
        return self
    
    def add_alternative_approach(self, approach: str) -> 'ExplanationBuilder':
        """Add an alternative approach that was considered"""
        self.alternative_approaches.append(approach)
        return self
    
    def add_limitation(self, limitation: str) -> 'ExplanationBuilder':
        """Add a known limitation"""
        self.limitations.append(limitation)
        return self
    
    def set_input_summary(self, input_data: Dict[str, Any]) -> 'ExplanationBuilder':
        """Set the input summary"""
        self.input_summary = input_data
        return self
    
    def set_output_summary(self, output_data: Dict[str, Any]) -> 'ExplanationBuilder':
        """Set the output summary"""
        self.output_summary = output_data
        return self
    
    def build_chain_of_thought(self, decision_context: str, final_reasoning: str, 
                              decision: str, confidence: float = 0.8) -> ChainOfThought:
        """Build a Chain-of-Thought explanation"""
        return ChainOfThought(
            agent_name=self.agent_name,
            decision_context=decision_context,
            thought_steps=self.thought_steps.copy(),
            final_reasoning=final_reasoning,
            decision=decision,
            confidence=confidence
        )
    
    def build_react_trace(self, task_context: str, final_outcome: str, 
                         success: bool = True) -> ReActTrace:
        """Build a ReAct trace explanation"""
        # Combine all steps in chronological order
        all_steps = []
        all_steps.extend(self.thought_steps)
        all_steps.extend(self.action_steps)
        all_steps.extend(self.observation_steps)
        
        # Sort by timestamp
        all_steps.sort(key=lambda x: x.timestamp)
        
        return ReActTrace(
            agent_name=self.agent_name,
            task_context=task_context,
            steps=all_steps,
            final_outcome=final_outcome,
            success=success,
            total_steps=len(all_steps)
        )
    
    def build_explanation(self, chain_of_thought: Optional[ChainOfThought] = None,
                         react_trace: Optional[ReActTrace] = None,
                         confidence_score: float = 0.8) -> AgentExplanation:
        """Build the final agent explanation"""
        processing_time = time.time() - self.start_time
        
        return AgentExplanation(
            agent_name=self.agent_name,
            task_description=self.task_description,
            input_summary=self.input_summary,
            output_summary=self.output_summary,
            chain_of_thought=chain_of_thought,
            react_trace=react_trace,
            decision_factors=self.decision_factors.copy(),
            alternative_approaches=self.alternative_approaches.copy(),
            limitations=self.limitations.copy(),
            confidence_score=confidence_score,
            processing_time=processing_time
        )


def create_explanation_builder(agent_name: str, task_description: str) -> ExplanationBuilder:
    """Factory function to create an explanation builder"""
    return ExplanationBuilder(agent_name, task_description)
