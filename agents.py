from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, Any, Optional, Tuple, List
import json
import time

from models import (
    KnowledgeGap, TopicPlan, TopicDetail, 
    UserInput, AgentState, AgentExplanation
)
from config import GOOGLE_API_KEY, MODEL_NAME, TEMPERATURE
from base_agent import BaseAgent

# Initialize Gemini model (only if API key is available)
llm = None
if GOOGLE_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=TEMPERATURE
        )
    except Exception as e:
        print(f"Warning: Could not initialize Gemini model: {e}")
        llm = None

class GapAnalysisAgent(BaseAgent):
    """First agent: Identifies knowledge gaps between user background and target topic"""
    
    def __init__(self):
        super().__init__("GapAnalysisAgent", "Identify knowledge gaps between user background and target topic")
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are an expert educational consultant specializing in gap analysis.
        
        Analyze the user's current background and the topic they want to learn to identify knowledge gaps.
        
        User Input:
        - Topic to learn: {topic}
        - Current background: {background}
        - Preferred learning format: {preferred_format}
        
        Please provide a comprehensive gap analysis in the following JSON format:
        {{
            "identified_gaps": ["gap1", "gap2", "gap3"],
            "current_level": "detailed assessment of current knowledge level",
            "target_level": "what knowledge level they need to achieve",
            "gap_analysis": "detailed explanation of the knowledge gaps and why they exist"
        }}
        
        Be specific and actionable. Consider the learning format preference when analyzing gaps.
        """)
        
        # Set LLM reference
        self.llm = llm
        
        # Create chain only if LLM is available
        if llm:
            self.chain = self.prompt | llm | JsonOutputParser()
        else:
            self.chain = None
    
    def analyze_gaps(self, user_input: UserInput) -> Tuple[KnowledgeGap, AgentExplanation]:
        """Analyze knowledge gaps between user background and target topic with explainability"""
        input_data = {
            "topic": user_input.topic,
            "background": user_input.background,
            "preferred_format": user_input.preferred_format.value
        }
        
        initial_thoughts = [
            ("I need to analyze the user's current background and compare it with the target topic to identify specific knowledge gaps.", 0.9),
            (f"The user wants to learn '{user_input.topic}' and has background: '{user_input.background[:100]}...'", 0.8),
            (f"Their preferred learning format is {user_input.preferred_format.value}, which will influence how I structure the gap analysis.", 0.9)
        ]
        
        result, explanation = self.execute_with_explanation(
            input_data=input_data,
            initial_thoughts=initial_thoughts
        )
        
        return result, explanation
    
    def _get_fallback_result(self) -> KnowledgeGap:
        """Get fallback result when LLM is not available"""
        return KnowledgeGap(
            identified_gaps=["Basic understanding needed"],
            current_level="Beginner",
            target_level="Intermediate",
            gap_analysis="General knowledge gap identified (LLM not available)"
        )
        
    def _get_fallback_description(self) -> str:
        """Get description for fallback result"""
        return "Identified basic understanding as primary gap"
    
    def _process_llm_result(self, result: Dict[str, Any]) -> KnowledgeGap:
        """Process and validate LLM result"""
        return KnowledgeGap(**result)
    
    def _get_success_description(self, result: KnowledgeGap) -> str:
        """Get success description for the result"""
        return f"Identified {len(result.identified_gaps)} knowledge gaps"

class TopicPlanningAgent(BaseAgent):
    """Second agent: Creates a comprehensive topic plan based on identified gaps"""
    
    def __init__(self):
        super().__init__("TopicPlanningAgent", "Create comprehensive topic plan based on identified knowledge gaps")
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are an expert curriculum designer specializing in creating learning plans.
        
        Based on the identified knowledge gaps, create a comprehensive topic plan.
        
        Context:
        - Topic: {topic}
        - Knowledge gaps: {gaps}
        - Current level: {current_level}
        - Target level: {target_level}
        - Preferred format: {preferred_format}
        
        Create a structured learning plan in this JSON format:
        {{
            "main_topics": ["topic1", "topic2", "topic3"],
            "subtopics": ["subtopic1", "subtopic2", "subtopic3"],
            "learning_objectives": ["objective1", "objective2", "objective3"],
            "estimated_duration": "estimated time to complete"
        }}
        
        IMPORTANT: 
        - main_topics must be an array of strings
        - subtopics must be an array of strings
        - learning_objectives must be an array of strings
        - estimated_duration must be a single string
        
        Ensure the plan addresses all identified gaps and is appropriate for the user's level.
        Consider the preferred learning format when structuring the plan.
        """)
        
        # Set LLM reference
        self.llm = llm
        
        # Create chain only if LLM is available
        if llm:
            self.chain = self.prompt | llm | JsonOutputParser()
        else:
            self.chain = None
    
    def create_plan(self, user_input: UserInput, knowledge_gap: KnowledgeGap) -> Tuple[TopicPlan, AgentExplanation]:
        """Create a comprehensive topic plan with explainability"""
        input_data = {
            "topic": user_input.topic,
            "gaps": knowledge_gap.identified_gaps,
            "current_level": knowledge_gap.current_level,
            "target_level": knowledge_gap.target_level,
            "preferred_format": user_input.preferred_format.value
        }
        
        initial_thoughts = [
            (f"I need to create a learning plan that addresses the identified gaps: {', '.join(knowledge_gap.identified_gaps)}", 0.9),
            (f"The user is at '{knowledge_gap.current_level}' level and needs to reach '{knowledge_gap.target_level}' level", 0.8),
            (f"Their preferred learning format is {user_input.preferred_format.value}, so I should structure the plan accordingly", 0.9)
        ]
        
        additional_factors = [
            "Identified knowledge gaps",
            "Current vs target knowledge level",
            "Educational progression principles",
            "Topic complexity and scope"
        ]
        
        result, explanation = self.execute_with_explanation(
            input_data=input_data,
            initial_thoughts=initial_thoughts,
            additional_factors=additional_factors
        )
        
        return result, explanation
    
    def _get_fallback_result(self) -> TopicPlan:
        """Get fallback result when LLM is not available"""
        return TopicPlan(
            main_topics=["Introduction", "Core Concepts", "Advanced Topics"],
            subtopics=["Basics", "Fundamentals", "Applications"],
            learning_objectives=["Understand basics", "Master fundamentals", "Apply knowledge"],
            estimated_duration="4-6 weeks (LLM not available)"
        )
    
    def _get_fallback_description(self) -> str:
        """Get description for fallback result"""
        return "Created generic structure covering basic progression"
    
    def _process_llm_result(self, result: Dict[str, Any]) -> TopicPlan:
        """Process and validate LLM result"""
        # Ensure all fields are the correct type
        if not isinstance(result.get("main_topics"), list):
            result["main_topics"] = ["Introduction", "Core Concepts", "Advanced Topics"]
        
        if not isinstance(result.get("subtopics"), list):
            result["subtopics"] = ["Basics", "Fundamentals", "Applications"]
        
        if not isinstance(result.get("learning_objectives"), list):
            result["learning_objectives"] = ["Understand basics", "Master fundamentals", "Apply knowledge"]
        
        if not isinstance(result.get("estimated_duration"), str):
            result["estimated_duration"] = "4-6 weeks"
        
        return TopicPlan(**result)
    
    def _get_success_description(self, result: TopicPlan) -> str:
        """Get success description for the result"""
        return f"Created plan with {len(result.main_topics)} main topics and {len(result.learning_objectives)} objectives"
    
    def _get_validation_steps(self, result: Dict[str, Any], processed_result: TopicPlan) -> List[str]:
        """Get validation steps for result processing"""
        steps = []
        if not isinstance(result.get("main_topics"), list):
            steps.append("Fixed main_topics field type")
        if not isinstance(result.get("subtopics"), list):
            steps.append("Fixed subtopics field type")
        if not isinstance(result.get("learning_objectives"), list):
            steps.append("Fixed learning_objectives field type")
        if not isinstance(result.get("estimated_duration"), str):
            steps.append("Fixed estimated_duration field type")
        return steps

class TopicDetailAgent(BaseAgent):
    """Third agent: Provides detailed breakdown of each topic with resources and exercises"""
    
    def __init__(self):
        super().__init__("TopicDetailAgent", "Create detailed breakdown for specific topics")
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are an expert learning content specialist who creates detailed topic breakdowns.
        
        Create detailed information for each topic in the learning plan.
        
        Context:
        - Topic name: {topic_name}
        - Main topic: {main_topic}
        - Learning objective: {objective}
        - Preferred format: {preferred_format}
        - User background: {background}
        
        Provide detailed breakdown in this JSON format:
        {{
            "topic_name": "exact topic name",
            "description": "comprehensive description of what this topic covers",
            "resources": ["resource1", "resource2", "resource3"],
            "exercises": ["exercise1", "exercise2", "exercise3"],
            "assessment_criteria": "how to assess understanding of this topic"
        }}
        
        IMPORTANT: 
        - topic_name must be a single string
        - description must be a single string
        - resources must be an array of strings (not objects)
        - exercises must be an array of strings (not objects)
        - assessment_criteria must be a single string
        
        Make resources and exercises specific to the preferred learning format.
        Ensure exercises are appropriate for the user's background level.
        """)
        
        # Set LLM reference
        self.llm = llm
        
        # Create chain only if LLM is available
        if llm:
            self.chain = self.prompt | llm | JsonOutputParser()
        else:
            self.chain = None
    
    def create_topic_detail(self, topic_name: str, main_topic: str, objective: str, 
                           user_input: UserInput) -> Tuple[TopicDetail, AgentExplanation]:
        """Create detailed breakdown for a specific topic with explainability"""
        input_data = {
            "topic_name": topic_name,
            "main_topic": main_topic,
            "objective": objective,
            "preferred_format": user_input.preferred_format.value,
            "background": user_input.background
        }
        
        initial_thoughts = [
            (f"I need to create detailed content for the topic '{topic_name}' which is part of the main topic '{main_topic}'", 0.9),
            (f"The learning objective is: '{objective}' and the user prefers {user_input.preferred_format.value} format", 0.8),
            (f"User background: '{user_input.background[:100]}...' - this will help me tailor the resources and exercises", 0.8)
        ]
        
        additional_factors = [
            "Specific topic name and scope",
            "Learning objective for this topic",
            "User's background knowledge level",
            "Main topic context"
        ]
        
        additional_limitations = [
            "Resources based on general knowledge, not real-time availability",
            "Exercises may need adjustment based on user's actual progress",
            "Assessment criteria are general guidelines"
        ]
        
        result, explanation = self.execute_with_explanation(
            input_data=input_data,
            initial_thoughts=initial_thoughts,
            additional_factors=additional_factors,
            additional_limitations=additional_limitations
        )
        
        return result, explanation
    
    def _get_fallback_result(self) -> TopicDetail:
        """Get fallback result when LLM is not available"""
        return TopicDetail(
            topic_name="Generic Topic",
            description="Comprehensive coverage of topic (LLM not available)",
            resources=["Online course", "Practice exercises", "Reference materials"],
            exercises=["Multiple choice questions", "Practical projects", "Self-assessment"],
            assessment_criteria="Demonstrate understanding through practical application"
        )
    
    def _get_fallback_description(self) -> str:
        """Get description for fallback result"""
        return "Created generic structure with standard resources and exercises"
    
    def _process_llm_result(self, result: Dict[str, Any]) -> TopicDetail:
        """Process and validate LLM result"""
        # Ensure all fields are the correct type
        if not isinstance(result.get("topic_name"), str):
            result["topic_name"] = result.get("topic_name", "Generic Topic")
        
        if not isinstance(result.get("description"), str):
            result["description"] = f"Comprehensive coverage of {result.get('topic_name', 'topic')}"
        
        # Ensure resources is a list of strings
        if isinstance(result.get("resources"), list):
            result["resources"] = [str(item) if not isinstance(item, str) else item for item in result["resources"]]
        else:
            result["resources"] = ["Online course", "Practice exercises", "Reference materials"]
        
        # Ensure exercises is a list of strings
        if isinstance(result.get("exercises"), list):
            result["exercises"] = [str(item) if not isinstance(item, str) else item for item in result["exercises"]]
        else:
            result["exercises"] = ["Multiple choice questions", "Practical projects", "Self-assessment"]
        
        if not isinstance(result.get("assessment_criteria"), str):
            result["assessment_criteria"] = "Demonstrate understanding through practical application"
        
        return TopicDetail(**result)
    
    def _get_success_description(self, result: TopicDetail) -> str:
        """Get success description for the result"""
        return f"Created detail with {len(result.resources)} resources and {len(result.exercises)} exercises"
    
    def _get_validation_steps(self, result: Dict[str, Any], processed_result: TopicDetail) -> List[str]:
        """Get validation steps for result processing"""
        steps = []
        if not isinstance(result.get("topic_name"), str):
            steps.append("Fixed topic_name field type")
        if not isinstance(result.get("description"), str):
            steps.append("Fixed description field type")
        if not isinstance(result.get("resources"), list):
            steps.append("Fixed resources field type")
        if not isinstance(result.get("exercises"), list):
            steps.append("Fixed exercises field type")
        if not isinstance(result.get("assessment_criteria"), str):
            steps.append("Fixed assessment_criteria field type")
        return steps

class PlanCombinerAgent(BaseAgent):
    """Agent to combine all components into a complete learning plan"""
    
    def __init__(self):
        super().__init__("PlanCombinerAgent", "Combine all learning components into a cohesive plan")
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are an expert learning coordinator who combines all learning components into a cohesive plan.
        
        Combine the gap analysis, topic plan, and topic details into a complete learning plan.
        
        Context:
        - User input: {user_input}
        - Knowledge gaps: {gaps}
        - Topic plan: {plan}
        - Topic details: {details}
        
        Create a comprehensive learning plan in this JSON format:
        {{
            "learning_path": "step-by-step learning path with clear progression as a single string",
            "recommended_resources": ["overall resource1", "overall resource2"],
            "timeline": "suggested timeline for completion as a single string",
            "success_metrics": ["metric1", "metric2", "metric3"]
        }}
        
        IMPORTANT: 
        - learning_path must be a single string, not a list
        - timeline must be a single string, not a list
        - recommended_resources must be an array of strings
        - success_metrics must be an array of strings
        
        Ensure the plan flows logically and addresses all identified gaps.
        Make it actionable and measurable for the user.
        """)
        
        # Set LLM reference
        self.llm = llm
        
        # Create chain only if LLM is available
        if llm:
            self.chain = self.prompt | llm | JsonOutputParser()
        else:
            self.chain = None
    
    def combine_plan(self, user_input: UserInput, knowledge_gap: KnowledgeGap, 
                    topic_plan: TopicPlan, topic_details: list[TopicDetail]) -> Tuple[Dict[str, Any], AgentExplanation]:
        """Combine all components into a complete learning plan with explainability"""
        input_data = {
            "user_input": user_input.model_dump(),
            "gaps": knowledge_gap.model_dump(),
            "plan": topic_plan.model_dump(),
            "details": [detail.model_dump() for detail in topic_details]
        }
        
        initial_thoughts = [
            (f"I need to combine the gap analysis, topic plan, and {len(topic_details)} topic details into a cohesive learning plan", 0.9),
            (f"The plan has {len(topic_plan.main_topics)} main topics and {len(topic_plan.learning_objectives)} objectives", 0.8),
            (f"I need to create a clear learning path that flows logically and addresses all identified gaps", 0.9)
        ]
        
        additional_factors = [
            "Identified knowledge gaps from analysis",
            "Main topics and learning objectives",
            "Detailed topic breakdowns and resources",
            "Educational progression principles"
        ]
        
        additional_limitations = [
            "Combination based on static analysis, not real-time progress",
            "Timeline estimates are general guidelines",
            "Success metrics are suggested, not guaranteed"
        ]
        
        result, explanation = self.execute_with_explanation(
            input_data=input_data,
            initial_thoughts=initial_thoughts,
            additional_factors=additional_factors,
            additional_limitations=additional_limitations
        )
        
        return result, explanation
    
    def _get_fallback_result(self) -> Dict[str, Any]:
        """Get fallback result when LLM is not available"""
        return {
            "learning_path": "Follow the structured topics in order, practice regularly, and assess progress (LLM not available)",
            "recommended_resources": ["Online courses", "Practice platforms", "Community forums"],
            "timeline": "4-6 weeks with 2-3 hours per week",
            "success_metrics": ["Complete all exercises", "Pass assessments", "Apply knowledge practically"]
        }
    
    def _get_fallback_description(self) -> str:
        """Get description for fallback result"""
        return "Created generic structure with standard progression"
    
    def _process_llm_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate LLM result"""
        # Ensure learning_path is a string, not a list
        if isinstance(result.get("learning_path"), list):
            result["learning_path"] = " → ".join(result["learning_path"])
        elif not isinstance(result.get("learning_path"), str):
            result["learning_path"] = "Follow the structured topics in order, practice regularly, and assess progress"
        
        # Ensure other fields are the correct type
        if not isinstance(result.get("timeline"), str):
            result["timeline"] = "4-6 weeks with 2-3 hours per week"
        
        if not isinstance(result.get("recommended_resources"), list):
            result["recommended_resources"] = ["Online courses", "Practice platforms", "Community forums"]
        
        if not isinstance(result.get("success_metrics"), list):
            result["success_metrics"] = ["Complete all exercises", "Pass assessments", "Apply knowledge practically"]
        
        return result
    
    def _get_success_description(self, result: Dict[str, Any]) -> str:
        """Get success description for the result"""
        return f"Created combined plan with {len(result['recommended_resources'])} resources and {len(result['success_metrics'])} metrics"
    
    def _get_validation_steps(self, result: Dict[str, Any], processed_result: Dict[str, Any]) -> List[str]:
        """Get validation steps for result processing"""
        steps = []
        if isinstance(result.get("learning_path"), list):
            steps.append("Fixed learning_path field type (converted list to string)")
        elif not isinstance(result.get("learning_path"), str):
            steps.append("Fixed learning_path field type with default")
        if not isinstance(result.get("timeline"), str):
            steps.append("Fixed timeline field type")
        if not isinstance(result.get("recommended_resources"), list):
            steps.append("Fixed recommended_resources field type")
        if not isinstance(result.get("success_metrics"), list):
            steps.append("Fixed success_metrics field type")
        return steps 