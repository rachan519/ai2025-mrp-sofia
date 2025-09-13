"""
Base Agent Class with Explainability Features
Provides common explainability functionality for all agents to reduce code duplication.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import time

from models import AgentExplanation
from explanation_logger import create_explanation_builder


class BaseAgent(ABC):
    """Base class for all agents with built-in explainability features"""
    
    def __init__(self, agent_name: str, task_description: str):
        self.agent_name = agent_name
        self.task_description = task_description
        self.llm = None  # Will be set by subclasses
    
    def _create_explanation_builder(self) -> Any:
        """Create explanation builder for this agent"""
        return create_explanation_builder(self.agent_name, self.task_description)
    
    def _add_common_decision_factors(self, explanation_builder: Any, 
                                   additional_factors: List[str] = None) -> None:
        """Add common decision factors to explanation"""
        common_factors = [
            "User's stated background knowledge",
            "Target topic complexity", 
            "Preferred learning format",
            "Educational best practices"
        ]
        
        if additional_factors:
            common_factors.extend(additional_factors)
        
        for factor in common_factors:
            explanation_builder.add_decision_factor(factor)
    
    def _add_common_alternative_approaches(self, explanation_builder: Any,
                                         additional_approaches: List[str] = None) -> None:
        """Add common alternative approaches to explanation"""
        common_approaches = [
            "Surface-level analysis without comprehensive consideration",
            "Generic approach without personalization",
            "Overly complex approach that might overwhelm the user"
        ]
        
        if additional_approaches:
            common_approaches.extend(additional_approaches)
        
        for approach in common_approaches:
            explanation_builder.add_alternative_approach(approach)
    
    def _add_common_limitations(self, explanation_builder: Any,
                              additional_limitations: List[str] = None) -> None:
        """Add common limitations to explanation"""
        common_limitations = [
            "Analysis based on self-reported background information",
            "May not capture implicit knowledge or skills",
            "Results depend on topic complexity assessment"
        ]
        
        if additional_limitations:
            common_limitations.extend(additional_limitations)
        
        for limitation in common_limitations:
            explanation_builder.add_limitation(limitation)
    
    def _handle_llm_unavailable(self, explanation_builder: Any, 
                              fallback_result: Any, 
                              fallback_description: str) -> Tuple[Any, AgentExplanation]:
        """Handle case when LLM is not available"""
        explanation_builder.add_action(
            "fallback_analysis",
            f"Using default {self.agent_name.lower()} due to LLM unavailability",
            {"llm_available": False},
            {"fallback_used": True},
            success=True
        )
        
        explanation_builder.add_observation(
            "llm_unavailable",
            f"LLM is not available, using default {self.agent_name.lower()}",
            "system",
            relevance_score=1.0
        )
        
        cot = explanation_builder.build_chain_of_thought(
            f"{self.agent_name} with limited information",
            f"Used default {self.agent_name.lower()} due to LLM unavailability. {fallback_description}",
            f"Provide generic {self.agent_name.lower()}",
            confidence=0.3
        )
        
        explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.3)
        return fallback_result, explanation
    
    def _handle_llm_error(self, explanation_builder: Any, error: Exception,
                         fallback_result: Any, fallback_description: str) -> Tuple[Any, AgentExplanation]:
        """Handle case when LLM encounters an error"""
        explanation_builder.add_action(
            "error_handling",
            f"Handling error in {self.agent_name.lower()}: {str(error)}",
            {"error": str(error)},
            {"error_handled": True},
            success=False,
            error_message=str(error)
        )
        
        explanation_builder.add_observation(
            "error_occurred",
            f"Error occurred during LLM {self.agent_name.lower()}: {str(error)}",
            "system",
            relevance_score=1.0
        )
        
        cot = explanation_builder.build_chain_of_thought(
            f"Error handling in {self.agent_name.lower()}",
            f"Encountered error during LLM {self.agent_name.lower()}: {str(error)}. {fallback_description}",
            f"Use default {self.agent_name.lower()} due to error",
            confidence=0.4
        )
        
        explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.4)
        return fallback_result, explanation
    
    def _add_llm_analysis_action(self, explanation_builder: Any, 
                                input_data: Dict[str, Any],
                                analysis_type: str) -> None:
        """Add LLM analysis action to explanation"""
        explanation_builder.add_action(
            "llm_analysis",
            f"Using LLM to perform {analysis_type}",
            input_data,
            {"analysis_type": analysis_type},
            success=True
        )
    
    def _add_llm_result_observation(self, explanation_builder: Any, 
                                  result: Dict[str, Any],
                                  result_description: str) -> None:
        """Add LLM result observation to explanation"""
        explanation_builder.add_observation(
            "llm_result",
            result_description,
            "llm",
            relevance_score=0.9
        )
    
    def _add_validation_thoughts(self, explanation_builder: Any, 
                               validation_steps: List[str]) -> None:
        """Add validation thoughts to explanation"""
        for step in validation_steps:
            explanation_builder.add_thought(step, confidence=0.8)
    
    def _build_successful_explanation(self, explanation_builder: Any,
                                    result: Dict[str, Any],
                                    success_description: str,
                                    confidence: float = 0.8) -> AgentExplanation:
        """Build explanation for successful LLM analysis"""
        explanation_builder.add_thought(
            success_description,
            confidence=confidence
        )
        
        cot = explanation_builder.build_chain_of_thought(
            f"Comprehensive {self.agent_name.lower()} using LLM",
            f"Successfully completed {self.task_description.lower()} using LLM analysis.",
            success_description,
            confidence=confidence
        )
        
        return explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=confidence)
    
    def _setup_explanation_builder(self, input_data: Dict[str, Any],
                                 additional_factors: List[str] = None,
                                 additional_approaches: List[str] = None,
                                 additional_limitations: List[str] = None) -> Any:
        """Setup explanation builder with common configuration"""
        explanation_builder = self._create_explanation_builder()
        explanation_builder.set_input_summary(input_data)
        
        # Add common decision factors, approaches, and limitations
        self._add_common_decision_factors(explanation_builder, additional_factors)
        self._add_common_alternative_approaches(explanation_builder, additional_approaches)
        self._add_common_limitations(explanation_builder, additional_limitations)
        
        return explanation_builder
    
    def _add_initial_thoughts(self, explanation_builder: Any, 
                            thoughts: List[str]) -> None:
        """Add initial thoughts to explanation"""
        for thought, confidence in thoughts:
            explanation_builder.add_thought(thought, confidence)
    
    @abstractmethod
    def _get_fallback_result(self) -> Any:
        """Get fallback result when LLM is not available"""
        pass
    
    @abstractmethod
    def _get_fallback_description(self) -> str:
        """Get description for fallback result"""
        pass
    
    @abstractmethod
    def _process_llm_result(self, result: Dict[str, Any]) -> Any:
        """Process and validate LLM result"""
        pass
    
    @abstractmethod
    def _get_success_description(self, result: Any) -> str:
        """Get success description for the result"""
        pass
    
    def execute_with_explanation(self, input_data: Dict[str, Any],
                               additional_factors: List[str] = None,
                               additional_approaches: List[str] = None,
                               additional_limitations: List[str] = None,
                               initial_thoughts: List[Tuple[str, float]] = None) -> Tuple[Any, AgentExplanation]:
        """Execute agent with full explainability - template method"""
        
        # Setup explanation builder
        explanation_builder = self._setup_explanation_builder(
            input_data, additional_factors, additional_approaches, additional_limitations
        )
        
        # Add initial thoughts
        if initial_thoughts:
            self._add_initial_thoughts(explanation_builder, initial_thoughts)
        
        # Handle LLM unavailable case
        if not self.llm:
            fallback_result = self._get_fallback_result()
            fallback_description = self._get_fallback_description()
            return self._handle_llm_unavailable(explanation_builder, fallback_result, fallback_description)
        
        try:
            # Add LLM analysis action
            self._add_llm_analysis_action(explanation_builder, input_data, f"{self.agent_name.lower()}")
            
            # Execute LLM chain
            result = self.chain.invoke(input_data)
            
            # Add LLM result observation
            self._add_llm_result_observation(explanation_builder, result, 
                                           f"LLM completed {self.agent_name.lower()} successfully")
            
            # Process and validate result
            processed_result = self._process_llm_result(result)
            
            # Add validation thoughts if needed
            validation_steps = self._get_validation_steps(result, processed_result)
            if validation_steps:
                self._add_validation_thoughts(explanation_builder, validation_steps)
            
            # Build successful explanation
            success_description = self._get_success_description(processed_result)
            explanation = self._build_successful_explanation(
                explanation_builder, processed_result, success_description
            )
            
            # Set output summary
            explanation_builder.set_output_summary(processed_result if isinstance(processed_result, dict) else processed_result.__dict__)
            
            return processed_result, explanation
            
        except Exception as e:
            # Handle LLM error
            fallback_result = self._get_fallback_result()
            fallback_description = self._get_fallback_description()
            return self._handle_llm_error(explanation_builder, e, fallback_result, fallback_description)
    
    def _get_validation_steps(self, result: Dict[str, Any], processed_result: Any) -> List[str]:
        """Get validation steps for result processing - override in subclasses if needed"""
        return []
