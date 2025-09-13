from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any, Union
from enum import Enum
from datetime import datetime

class LearningFormat(str, Enum):
    VIDEO = "video"
    TEXT = "text"
    AUDIO = "audio"

class UserInput(BaseModel):
    topic: str = Field(..., description="The topic the user wants to learn")
    background: str = Field(..., description="User's current background and knowledge level")
    preferred_format: LearningFormat = Field(..., description="Preferred learning format")
    max_iterations: Optional[int] = Field(default=5, description="Maximum number of iterations")

class KnowledgeGap(BaseModel):
    identified_gaps: List[str] = Field(..., description="List of knowledge gaps identified")
    current_level: str = Field(..., description="Current knowledge level assessment")
    target_level: str = Field(..., description="Target knowledge level for the topic")
    gap_analysis: str = Field(..., description="Detailed analysis of the knowledge gap")

class TopicPlan(BaseModel):
    main_topics: List[str] = Field(..., description="Main topics to cover")
    subtopics: List[str] = Field(..., description="Subtopics for each main topic")
    learning_objectives: List[str] = Field(..., description="Learning objectives")
    estimated_duration: str = Field(..., description="Estimated time to complete")

class TopicDetail(BaseModel):
    topic_name: str = Field(..., description="Name of the topic")
    description: str = Field(..., description="Detailed description")
    resources: List[str] = Field(..., description="Recommended resources")
    exercises: List[str] = Field(..., description="Practice exercises")
    assessment_criteria: str = Field(..., description="How to assess understanding")

class CompleteLearningPlan(BaseModel):
    user_input: UserInput
    knowledge_gap: KnowledgeGap
    topic_plan: TopicPlan
    topic_details: List[TopicDetail]
    learning_path: str = Field(..., description="Step-by-step learning path")
    recommended_resources: List[str] = Field(..., description="Overall recommended resources")
    timeline: str = Field(..., description="Suggested timeline for completion")
    success_metrics: List[str] = Field(..., description="Metrics to measure progress")

# Explainability Models

class ThoughtStep(BaseModel):
    """A single thought step in Chain-of-Thought reasoning"""
    step_number: int = Field(..., description="Sequential number of this thought step")
    thought: str = Field(..., description="The reasoning or thought process")
    confidence: float = Field(..., description="Confidence level (0.0 to 1.0)", ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now, description="When this thought occurred")

class ActionStep(BaseModel):
    """A single action step in ReAct pattern"""
    action_type: str = Field(..., description="Type of action taken (e.g., 'analyze', 'generate', 'validate')")
    action_description: str = Field(..., description="Description of what action was performed")
    input_data: Dict[str, Any] = Field(..., description="Input data used for this action")
    output_data: Dict[str, Any] = Field(..., description="Output data produced by this action")
    success: bool = Field(..., description="Whether the action was successful")
    error_message: Optional[str] = Field(None, description="Error message if action failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this action occurred")

class ObservationStep(BaseModel):
    """An observation step in ReAct pattern"""
    observation_type: str = Field(..., description="Type of observation (e.g., 'result', 'feedback', 'validation')")
    observation_content: str = Field(..., description="Content of the observation")
    source: str = Field(..., description="Source of the observation (e.g., 'llm', 'validation', 'user')")
    relevance_score: float = Field(..., description="Relevance score (0.0 to 1.0)", ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now, description="When this observation occurred")

class ChainOfThought(BaseModel):
    """Complete Chain-of-Thought reasoning for an agent decision"""
    agent_name: str = Field(..., description="Name of the agent performing the reasoning")
    decision_context: str = Field(..., description="Context of the decision being made")
    thought_steps: List[ThoughtStep] = Field(..., description="Sequential thought steps")
    final_reasoning: str = Field(..., description="Final reasoning summary")
    decision: str = Field(..., description="The final decision made")
    confidence: float = Field(..., description="Overall confidence in the decision", ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now, description="When this reasoning occurred")

class ReActTrace(BaseModel):
    """Complete ReAct trace for an agent's reasoning and actions"""
    agent_name: str = Field(..., description="Name of the agent performing the reasoning")
    task_context: str = Field(..., description="Context of the task being performed")
    steps: List[Union[ThoughtStep, ActionStep, ObservationStep]] = Field(..., description="Mixed sequence of thoughts, actions, and observations")
    final_outcome: str = Field(..., description="Final outcome of the ReAct process")
    success: bool = Field(..., description="Whether the overall process was successful")
    total_steps: int = Field(..., description="Total number of steps taken")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this trace was completed")

class AgentExplanation(BaseModel):
    """Complete explanation for an agent's decision and actions"""
    agent_name: str = Field(..., description="Name of the agent")
    task_description: str = Field(..., description="Description of the task performed")
    input_summary: Dict[str, Any] = Field(..., description="Summary of inputs received")
    output_summary: Dict[str, Any] = Field(..., description="Summary of outputs produced")
    chain_of_thought: Optional[ChainOfThought] = Field(None, description="Chain-of-thought reasoning")
    react_trace: Optional[ReActTrace] = Field(None, description="ReAct trace if applicable")
    decision_factors: List[str] = Field(..., description="Key factors that influenced the decision")
    alternative_approaches: List[str] = Field(..., description="Alternative approaches considered")
    limitations: List[str] = Field(..., description="Known limitations of the approach")
    confidence_score: float = Field(..., description="Overall confidence score", ge=0.0, le=1.0)
    processing_time: float = Field(..., description="Time taken to process in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this explanation was created")

class AgentState(BaseModel):
    user_input: UserInput
    knowledge_gap: Optional[KnowledgeGap] = None
    topic_plan: Optional[TopicPlan] = None
    topic_details: Optional[List[TopicDetail]] = None
    complete_plan: Optional[CompleteLearningPlan] = None
    current_step: str = "gap_analysis"
    iteration_count: int = 0
    max_iterations: int = 5
    explanations: List[AgentExplanation] = Field(default_factory=list, description="Explanations for each agent's decisions") 