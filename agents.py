from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, Any, Optional, Tuple
import json
import time

from models import (
    KnowledgeGap, TopicPlan, TopicDetail, 
    UserInput, AgentState, AgentExplanation
)
from config import GOOGLE_API_KEY, MODEL_NAME, TEMPERATURE
from explanation_logger import ExplanationBuilder, create_explanation_builder

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

class GapAnalysisAgent:
    """First agent: Identifies knowledge gaps between user background and target topic"""
    
    def __init__(self):
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
        
        # Create chain only if LLM is available
        if llm:
            self.chain = self.prompt | llm | JsonOutputParser()
        else:
            self.chain = None
    
    def analyze_gaps(self, user_input: UserInput) -> Tuple[KnowledgeGap, AgentExplanation]:
        """Analyze knowledge gaps between user background and target topic with explainability"""
        # Create explanation builder
        explanation_builder = create_explanation_builder(
            "GapAnalysisAgent", 
            "Identify knowledge gaps between user background and target topic"
        )
        
        # Set input summary
        explanation_builder.set_input_summary({
            "topic": user_input.topic,
            "background": user_input.background,
            "preferred_format": user_input.preferred_format.value
        })
        
        # Add initial thoughts
        explanation_builder.add_thought(
            "I need to analyze the user's current background and compare it with the target topic to identify specific knowledge gaps.",
            confidence=0.9
        )
        
        explanation_builder.add_thought(
            f"The user wants to learn '{user_input.topic}' and has background: '{user_input.background[:100]}...'",
            confidence=0.8
        )
        
        explanation_builder.add_thought(
            f"Their preferred learning format is {user_input.preferred_format.value}, which will influence how I structure the gap analysis.",
            confidence=0.9
        )
        
        # Add decision factors
        explanation_builder.add_decision_factor("User's stated background knowledge")
        explanation_builder.add_decision_factor("Target topic complexity")
        explanation_builder.add_decision_factor("Preferred learning format")
        explanation_builder.add_decision_factor("Educational best practices for gap analysis")
        
        # Add alternative approaches
        explanation_builder.add_alternative_approach("Surface-level gap analysis based only on topic keywords")
        explanation_builder.add_alternative_approach("Generic gap analysis without considering learning format")
        explanation_builder.add_alternative_approach("Overly detailed analysis that might overwhelm the user")
        
        # Add limitations
        explanation_builder.add_limitation("Analysis based only on user's self-reported background")
        explanation_builder.add_limitation("May not capture implicit knowledge or skills")
        explanation_builder.add_limitation("Gap identification depends on topic complexity assessment")
        
        if not llm:
            # Add action for fallback case
            explanation_builder.add_action(
                "fallback_analysis",
                "Using default gap analysis due to LLM unavailability",
                {"llm_available": False},
                {"fallback_used": True},
                success=True
            )
            
            # Add observation
            explanation_builder.add_observation(
                "llm_unavailable",
                "LLM is not available, using default gap analysis",
                "system",
                relevance_score=1.0
            )
            
            # Build explanation
            cot = explanation_builder.build_chain_of_thought(
                "Gap analysis with limited information",
                "Used default analysis due to LLM unavailability. Identified basic understanding as primary gap.",
                "Provide generic gap analysis",
                confidence=0.3
            )
            
            explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.3)
            
            # Set output summary
            explanation_builder.set_output_summary({
                "identified_gaps": ["Basic understanding needed"],
                "current_level": "Beginner",
                "target_level": "Intermediate"
            })
            
            return KnowledgeGap(
                identified_gaps=["Basic understanding needed"],
                current_level="Beginner",
                target_level="Intermediate",
                gap_analysis="General knowledge gap identified (LLM not available)"
            ), explanation
        
        try:
            # Add action for LLM analysis
            explanation_builder.add_action(
                "llm_analysis",
                "Using LLM to perform comprehensive gap analysis",
                {
                    "topic": user_input.topic,
                    "background": user_input.background,
                    "preferred_format": user_input.preferred_format.value
                },
                {"analysis_type": "comprehensive"},
                success=True
            )
            
            result = self.chain.invoke({
                "topic": user_input.topic,
                "background": user_input.background,
                "preferred_format": user_input.preferred_format.value
            })
            
            # Add observation of LLM result
            explanation_builder.add_observation(
                "llm_result",
                f"LLM identified {len(result.get('identified_gaps', []))} knowledge gaps",
                "llm",
                relevance_score=0.9
            )
            
            # Add final thoughts
            explanation_builder.add_thought(
                f"Successfully identified {len(result.get('identified_gaps', []))} specific knowledge gaps",
                confidence=0.8
            )
            
            # Build explanation
            cot = explanation_builder.build_chain_of_thought(
                "Comprehensive gap analysis using LLM",
                f"Analyzed user background against target topic '{user_input.topic}' and identified specific gaps using educational expertise.",
                f"Identified {len(result.get('identified_gaps', []))} knowledge gaps",
                confidence=0.8
            )
            
            explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.8)
            
            # Set output summary
            explanation_builder.set_output_summary(result)
            
            return KnowledgeGap(**result), explanation
            
        except Exception as e:
            # Add action for error case
            explanation_builder.add_action(
                "error_handling",
                f"Handling error in gap analysis: {str(e)}",
                {"error": str(e)},
                {"error_handled": True},
                success=False,
                error_message=str(e)
            )
            
            # Add observation of error
            explanation_builder.add_observation(
                "error_occurred",
                f"Error occurred during LLM analysis: {str(e)}",
                "system",
                relevance_score=1.0
            )
            
            print(f"Error in gap analysis: {e}")
            
            # Build explanation for error case
            cot = explanation_builder.build_chain_of_thought(
                "Error handling in gap analysis",
                f"Encountered error during LLM analysis: {str(e)}. Falling back to default analysis.",
                "Use default gap analysis due to error",
                confidence=0.4
            )
            
            explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.4)
            
            # Set output summary for error case
            explanation_builder.set_output_summary({
                "identified_gaps": ["Basic understanding needed"],
                "current_level": "Beginner",
                "target_level": "Intermediate"
            })
            
            return KnowledgeGap(
                identified_gaps=["Basic understanding needed"],
                current_level="Beginner",
                target_level="Intermediate",
                gap_analysis="General knowledge gap identified"
            ), explanation

class TopicPlanningAgent:
    """Second agent: Creates a comprehensive topic plan based on identified gaps"""
    
    def __init__(self):
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
        
        # Create chain only if LLM is available
        if llm:
            self.chain = self.prompt | llm | JsonOutputParser()
        else:
            self.chain = None
    
    def create_plan(self, user_input: UserInput, knowledge_gap: KnowledgeGap) -> Tuple[TopicPlan, AgentExplanation]:
        """Create a comprehensive topic plan with explainability"""
        # Create explanation builder
        explanation_builder = create_explanation_builder(
            "TopicPlanningAgent", 
            "Create comprehensive topic plan based on identified knowledge gaps"
        )
        
        # Set input summary
        explanation_builder.set_input_summary({
            "topic": user_input.topic,
            "gaps": knowledge_gap.identified_gaps,
            "current_level": knowledge_gap.current_level,
            "target_level": knowledge_gap.target_level,
            "preferred_format": user_input.preferred_format.value
        })
        
        # Add initial thoughts
        explanation_builder.add_thought(
            f"I need to create a learning plan that addresses the identified gaps: {', '.join(knowledge_gap.identified_gaps)}",
            confidence=0.9
        )
        
        explanation_builder.add_thought(
            f"The user is at '{knowledge_gap.current_level}' level and needs to reach '{knowledge_gap.target_level}' level",
            confidence=0.8
        )
        
        explanation_builder.add_thought(
            f"Their preferred learning format is {user_input.preferred_format.value}, so I should structure the plan accordingly",
            confidence=0.9
        )
        
        # Add decision factors
        explanation_builder.add_decision_factor("Identified knowledge gaps")
        explanation_builder.add_decision_factor("Current vs target knowledge level")
        explanation_builder.add_decision_factor("Preferred learning format")
        explanation_builder.add_decision_factor("Educational progression principles")
        explanation_builder.add_decision_factor("Topic complexity and scope")
        
        # Add alternative approaches
        explanation_builder.add_alternative_approach("Linear progression without considering gaps")
        explanation_builder.add_alternative_approach("Overly complex plan that might overwhelm the user")
        explanation_builder.add_alternative_approach("Generic plan without format consideration")
        
        # Add limitations
        explanation_builder.add_limitation("Plan based on self-reported background and gaps")
        explanation_builder.add_limitation("Estimated duration may vary based on individual learning pace")
        explanation_builder.add_limitation("Plan assumes standard learning progression")
        
        if not llm:
            # Add action for fallback case
            explanation_builder.add_action(
                "fallback_planning",
                "Using default topic plan due to LLM unavailability",
                {"llm_available": False},
                {"fallback_used": True},
                success=True
            )
            
            # Add observation
            explanation_builder.add_observation(
                "llm_unavailable",
                "LLM is not available, using default topic plan",
                "system",
                relevance_score=1.0
            )
            
            # Build explanation
            cot = explanation_builder.build_chain_of_thought(
                "Topic planning with limited information",
                "Used default plan due to LLM unavailability. Created generic structure covering basic progression.",
                "Provide generic topic plan",
                confidence=0.3
            )
            
            explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.3)
            
            # Set output summary
            explanation_builder.set_output_summary({
                "main_topics": ["Introduction", "Core Concepts", "Advanced Topics"],
                "subtopics": ["Basics", "Fundamentals", "Applications"],
                "learning_objectives": ["Understand basics", "Master fundamentals", "Apply knowledge"],
                "estimated_duration": "4-6 weeks (LLM not available)"
            })
            
            return TopicPlan(
                main_topics=["Introduction", "Core Concepts", "Advanced Topics"],
                subtopics=["Basics", "Fundamentals", "Applications"],
                learning_objectives=["Understand basics", "Master fundamentals", "Apply knowledge"],
                estimated_duration="4-6 weeks (LLM not available)"
            ), explanation
        
        try:
            # Add action for LLM planning
            explanation_builder.add_action(
                "llm_planning",
                "Using LLM to create comprehensive topic plan",
                {
                    "topic": user_input.topic,
                    "gaps": knowledge_gap.identified_gaps,
                    "current_level": knowledge_gap.current_level,
                    "target_level": knowledge_gap.target_level,
                    "preferred_format": user_input.preferred_format.value
                },
                {"planning_type": "comprehensive"},
                success=True
            )
            
            result = self.chain.invoke({
                "topic": user_input.topic,
                "gaps": knowledge_gap.identified_gaps,
                "current_level": knowledge_gap.current_level,
                "target_level": knowledge_gap.target_level,
                "preferred_format": user_input.preferred_format.value
            })
            
            # Add observation of LLM result
            explanation_builder.add_observation(
                "llm_result",
                f"LLM created plan with {len(result.get('main_topics', []))} main topics and {len(result.get('learning_objectives', []))} objectives",
                "llm",
                relevance_score=0.9
            )
            
            # Add validation thoughts
            explanation_builder.add_thought(
                "I need to validate that the LLM output has the correct data types for all fields",
                confidence=0.8
            )
            
            # Ensure all fields are the correct type
            if not isinstance(result.get("main_topics"), list):
                result["main_topics"] = ["Introduction", "Core Concepts", "Advanced Topics"]
                explanation_builder.add_thought("Fixed main_topics field type", confidence=0.9)
            
            if not isinstance(result.get("subtopics"), list):
                result["subtopics"] = ["Basics", "Fundamentals", "Applications"]
                explanation_builder.add_thought("Fixed subtopics field type", confidence=0.9)
            
            if not isinstance(result.get("learning_objectives"), list):
                result["learning_objectives"] = ["Understand basics", "Master fundamentals", "Apply knowledge"]
                explanation_builder.add_thought("Fixed learning_objectives field type", confidence=0.9)
            
            if not isinstance(result.get("estimated_duration"), str):
                result["estimated_duration"] = "4-6 weeks"
                explanation_builder.add_thought("Fixed estimated_duration field type", confidence=0.9)
            
            # Add final thoughts
            explanation_builder.add_thought(
                f"Successfully created a comprehensive plan with {len(result['main_topics'])} main topics",
                confidence=0.8
            )
            
            # Build explanation
            cot = explanation_builder.build_chain_of_thought(
                "Comprehensive topic planning using LLM",
                f"Created structured learning plan for '{user_input.topic}' addressing {len(knowledge_gap.identified_gaps)} identified gaps with format-specific considerations.",
                f"Created plan with {len(result['main_topics'])} main topics and {len(result['learning_objectives'])} objectives",
                confidence=0.8
            )
            
            explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.8)
            
            # Set output summary
            explanation_builder.set_output_summary(result)
            
            return TopicPlan(**result), explanation
            
        except Exception as e:
            # Add action for error case
            explanation_builder.add_action(
                "error_handling",
                f"Handling error in topic planning: {str(e)}",
                {"error": str(e)},
                {"error_handled": True},
                success=False,
                error_message=str(e)
            )
            
            # Add observation of error
            explanation_builder.add_observation(
                "error_occurred",
                f"Error occurred during LLM planning: {str(e)}",
                "system",
                relevance_score=1.0
            )
            
            print(f"Error in topic planning: {e}")
            
            # Build explanation for error case
            cot = explanation_builder.build_chain_of_thought(
                "Error handling in topic planning",
                f"Encountered error during LLM planning: {str(e)}. Falling back to default plan.",
                "Use default topic plan due to error",
                confidence=0.4
            )
            
            explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.4)
            
            # Set output summary for error case
            explanation_builder.set_output_summary({
                "main_topics": ["Introduction", "Core Concepts", "Advanced Topics"],
                "subtopics": ["Basics", "Fundamentals", "Applications"],
                "learning_objectives": ["Understand basics", "Master fundamentals", "Apply knowledge"],
                "estimated_duration": "4-6 weeks"
            })
            
            return TopicPlan(
                main_topics=["Introduction", "Core Concepts", "Advanced Topics"],
                subtopics=["Basics", "Fundamentals", "Applications"],
                learning_objectives=["Understand basics", "Master fundamentals", "Apply knowledge"],
                estimated_duration="4-6 weeks"
            ), explanation

class TopicDetailAgent:
    """Third agent: Provides detailed breakdown of each topic with resources and exercises"""
    
    def __init__(self):
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
        
        # Create chain only if LLM is available
        if llm:
            self.chain = self.prompt | llm | JsonOutputParser()
        else:
            self.chain = None
    
    def create_topic_detail(self, topic_name: str, main_topic: str, objective: str, 
                           user_input: UserInput) -> Tuple[TopicDetail, AgentExplanation]:
        """Create detailed breakdown for a specific topic with explainability"""
        # Create explanation builder
        explanation_builder = create_explanation_builder(
            "TopicDetailAgent", 
            f"Create detailed breakdown for topic: {topic_name}"
        )
        
        # Set input summary
        explanation_builder.set_input_summary({
            "topic_name": topic_name,
            "main_topic": main_topic,
            "objective": objective,
            "preferred_format": user_input.preferred_format.value,
            "background": user_input.background
        })
        
        # Add initial thoughts
        explanation_builder.add_thought(
            f"I need to create detailed content for the topic '{topic_name}' which is part of the main topic '{main_topic}'",
            confidence=0.9
        )
        
        explanation_builder.add_thought(
            f"The learning objective is: '{objective}' and the user prefers {user_input.preferred_format.value} format",
            confidence=0.8
        )
        
        explanation_builder.add_thought(
            f"User background: '{user_input.background[:100]}...' - this will help me tailor the resources and exercises",
            confidence=0.8
        )
        
        # Add decision factors
        explanation_builder.add_decision_factor("Specific topic name and scope")
        explanation_builder.add_decision_factor("Learning objective for this topic")
        explanation_builder.add_decision_factor("User's preferred learning format")
        explanation_builder.add_decision_factor("User's background knowledge level")
        explanation_builder.add_decision_factor("Main topic context")
        
        # Add alternative approaches
        explanation_builder.add_alternative_approach("Generic resources without format consideration")
        explanation_builder.add_alternative_approach("Overly complex exercises for the user's level")
        explanation_builder.add_alternative_approach("Vague assessment criteria")
        
        # Add limitations
        explanation_builder.add_limitation("Resources based on general knowledge, not real-time availability")
        explanation_builder.add_limitation("Exercises may need adjustment based on user's actual progress")
        explanation_builder.add_limitation("Assessment criteria are general guidelines")
        
        if not llm:
            # Add action for fallback case
            explanation_builder.add_action(
                "fallback_detailing",
                "Using default topic detail due to LLM unavailability",
                {"llm_available": False},
                {"fallback_used": True},
                success=True
            )
            
            # Add observation
            explanation_builder.add_observation(
                "llm_unavailable",
                "LLM is not available, using default topic detail",
                "system",
                relevance_score=1.0
            )
            
            # Build explanation
            cot = explanation_builder.build_chain_of_thought(
                "Topic detailing with limited information",
                "Used default detail due to LLM unavailability. Created generic structure with standard resources and exercises.",
                "Provide generic topic detail",
                confidence=0.3
            )
            
            explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.3)
            
            # Set output summary
            explanation_builder.set_output_summary({
                "topic_name": topic_name,
                "description": f"Comprehensive coverage of {topic_name} (LLM not available)",
                "resources": ["Online course", "Practice exercises", "Reference materials"],
                "exercises": ["Multiple choice questions", "Practical projects", "Self-assessment"],
                "assessment_criteria": "Demonstrate understanding through practical application"
            })
            
            return TopicDetail(
                topic_name=topic_name,
                description=f"Comprehensive coverage of {topic_name} (LLM not available)",
                resources=["Online course", "Practice exercises", "Reference materials"],
                exercises=["Multiple choice questions", "Practical projects", "Self-assessment"],
                assessment_criteria="Demonstrate understanding through practical application"
            ), explanation
        
        try:
            # Add action for LLM detailing
            explanation_builder.add_action(
                "llm_detailing",
                "Using LLM to create detailed topic breakdown",
                {
                    "topic_name": topic_name,
                    "main_topic": main_topic,
                    "objective": objective,
                    "preferred_format": user_input.preferred_format.value,
                    "background": user_input.background
                },
                {"detailing_type": "comprehensive"},
                success=True
            )
            
            result = self.chain.invoke({
                "topic_name": topic_name,
                "main_topic": main_topic,
                "objective": objective,
                "preferred_format": user_input.preferred_format.value,
                "background": user_input.background
            })
            
            # Add observation of LLM result
            explanation_builder.add_observation(
                "llm_result",
                f"LLM created detail with {len(result.get('resources', []))} resources and {len(result.get('exercises', []))} exercises",
                "llm",
                relevance_score=0.9
            )
            
            # Add validation thoughts
            explanation_builder.add_thought(
                "I need to validate and fix the data types for all fields to ensure consistency",
                confidence=0.8
            )
            
            # Ensure all fields are the correct type
            if not isinstance(result.get("topic_name"), str):
                result["topic_name"] = topic_name
                explanation_builder.add_thought("Fixed topic_name field type", confidence=0.9)
            
            if not isinstance(result.get("description"), str):
                result["description"] = f"Comprehensive coverage of {topic_name}"
                explanation_builder.add_thought("Fixed description field type", confidence=0.9)
            
            # Ensure resources is a list of strings
            if isinstance(result.get("resources"), list):
                result["resources"] = [str(item) if not isinstance(item, str) else item for item in result["resources"]]
                explanation_builder.add_thought("Validated and fixed resources field type", confidence=0.9)
            else:
                result["resources"] = ["Online course", "Practice exercises", "Reference materials"]
                explanation_builder.add_thought("Fixed resources field type with defaults", confidence=0.9)
            
            # Ensure exercises is a list of strings
            if isinstance(result.get("exercises"), list):
                result["exercises"] = [str(item) if not isinstance(item, str) else item for item in result["exercises"]]
                explanation_builder.add_thought("Validated and fixed exercises field type", confidence=0.9)
            else:
                result["exercises"] = ["Multiple choice questions", "Practical projects", "Self-assessment"]
                explanation_builder.add_thought("Fixed exercises field type with defaults", confidence=0.9)
            
            if not isinstance(result.get("assessment_criteria"), str):
                result["assessment_criteria"] = "Demonstrate understanding through practical application"
                explanation_builder.add_thought("Fixed assessment_criteria field type", confidence=0.9)
            
            # Add final thoughts
            explanation_builder.add_thought(
                f"Successfully created detailed breakdown with {len(result['resources'])} resources and {len(result['exercises'])} exercises",
                confidence=0.8
            )
            
            # Build explanation
            cot = explanation_builder.build_chain_of_thought(
                "Comprehensive topic detailing using LLM",
                f"Created detailed breakdown for '{topic_name}' with format-specific resources and exercises tailored to user background.",
                f"Created detail with {len(result['resources'])} resources and {len(result['exercises'])} exercises",
                confidence=0.8
            )
            
            explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.8)
            
            # Set output summary
            explanation_builder.set_output_summary(result)
            
            return TopicDetail(**result), explanation
            
        except Exception as e:
            # Add action for error case
            explanation_builder.add_action(
                "error_handling",
                f"Handling error in topic detailing: {str(e)}",
                {"error": str(e)},
                {"error_handled": True},
                success=False,
                error_message=str(e)
            )
            
            # Add observation of error
            explanation_builder.add_observation(
                "error_occurred",
                f"Error occurred during LLM detailing: {str(e)}",
                "system",
                relevance_score=1.0
            )
            
            print(f"Error in topic detailing: {e}")
            
            # Build explanation for error case
            cot = explanation_builder.build_chain_of_thought(
                "Error handling in topic detailing",
                f"Encountered error during LLM detailing: {str(e)}. Falling back to default detail.",
                "Use default topic detail due to error",
                confidence=0.4
            )
            
            explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.4)
            
            # Set output summary for error case
            explanation_builder.set_output_summary({
                "topic_name": topic_name,
                "description": f"Comprehensive coverage of {topic_name}",
                "resources": ["Online course", "Practice exercises", "Reference materials"],
                "exercises": ["Multiple choice questions", "Practical projects", "Self-assessment"],
                "assessment_criteria": "Demonstrate understanding through practical application"
            })
            
            return TopicDetail(
                topic_name=topic_name,
                description=f"Comprehensive coverage of {topic_name}",
                resources=["Online course", "Practice exercises", "Reference materials"],
                exercises=["Multiple choice questions", "Practical projects", "Self-assessment"],
                assessment_criteria="Demonstrate understanding through practical application"
            ), explanation

class PlanCombinerAgent:
    """Agent to combine all components into a complete learning plan"""
    
    def __init__(self):
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
        
        # Create chain only if LLM is available
        if llm:
            self.chain = self.prompt | llm | JsonOutputParser()
        else:
            self.chain = None
    
    def combine_plan(self, user_input: UserInput, knowledge_gap: KnowledgeGap, 
                    topic_plan: TopicPlan, topic_details: list[TopicDetail]) -> Tuple[Dict[str, Any], AgentExplanation]:
        """Combine all components into a complete learning plan with explainability"""
        # Create explanation builder
        explanation_builder = create_explanation_builder(
            "PlanCombinerAgent", 
            "Combine all learning components into a cohesive plan"
        )
        
        # Set input summary
        explanation_builder.set_input_summary({
            "user_input": user_input.model_dump(),
            "gaps": knowledge_gap.model_dump(),
            "plan": topic_plan.model_dump(),
            "details": [detail.model_dump() for detail in topic_details]
        })
        
        # Add initial thoughts
        explanation_builder.add_thought(
            f"I need to combine the gap analysis, topic plan, and {len(topic_details)} topic details into a cohesive learning plan",
            confidence=0.9
        )
        
        explanation_builder.add_thought(
            f"The plan has {len(topic_plan.main_topics)} main topics and {len(topic_plan.learning_objectives)} objectives",
            confidence=0.8
        )
        
        explanation_builder.add_thought(
            f"I need to create a clear learning path that flows logically and addresses all identified gaps",
            confidence=0.9
        )
        
        # Add decision factors
        explanation_builder.add_decision_factor("Identified knowledge gaps from analysis")
        explanation_builder.add_decision_factor("Main topics and learning objectives")
        explanation_builder.add_decision_factor("Detailed topic breakdowns and resources")
        explanation_builder.add_decision_factor("User's preferred learning format")
        explanation_builder.add_decision_factor("Educational progression principles")
        
        # Add alternative approaches
        explanation_builder.add_alternative_approach("Linear combination without considering gaps")
        explanation_builder.add_alternative_approach("Overly complex plan that might overwhelm the user")
        explanation_builder.add_alternative_approach("Generic combination without personalization")
        
        # Add limitations
        explanation_builder.add_limitation("Combination based on static analysis, not real-time progress")
        explanation_builder.add_limitation("Timeline estimates are general guidelines")
        explanation_builder.add_limitation("Success metrics are suggested, not guaranteed")
        
        if not llm:
            # Add action for fallback case
            explanation_builder.add_action(
                "fallback_combination",
                "Using default plan combination due to LLM unavailability",
                {"llm_available": False},
                {"fallback_used": True},
                success=True
            )
            
            # Add observation
            explanation_builder.add_observation(
                "llm_unavailable",
                "LLM is not available, using default plan combination",
                "system",
                relevance_score=1.0
            )
            
            # Build explanation
            cot = explanation_builder.build_chain_of_thought(
                "Plan combination with limited information",
                "Used default combination due to LLM unavailability. Created generic structure with standard progression.",
                "Provide generic combined plan",
                confidence=0.3
            )
            
            explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.3)
            
            # Set output summary
            explanation_builder.set_output_summary({
                "learning_path": "Follow the structured topics in order, practice regularly, and assess progress (LLM not available)",
                "recommended_resources": ["Online courses", "Practice platforms", "Community forums"],
                "timeline": "4-6 weeks with 2-3 hours per week",
                "success_metrics": ["Complete all exercises", "Pass assessments", "Apply knowledge practically"]
            })
            
            return {
                "learning_path": "Follow the structured topics in order, practice regularly, and assess progress (LLM not available)",
                "recommended_resources": ["Online courses", "Practice platforms", "Community forums"],
                "timeline": "4-6 weeks with 2-3 hours per week",
                "success_metrics": ["Complete all exercises", "Pass assessments", "Apply knowledge practically"]
            }, explanation
        
        try:
            # Add action for LLM combination
            explanation_builder.add_action(
                "llm_combination",
                "Using LLM to combine all components into cohesive plan",
                {
                    "user_input": user_input.model_dump(),
                    "gaps": knowledge_gap.model_dump(),
                    "plan": topic_plan.model_dump(),
                    "details": [detail.model_dump() for detail in topic_details]
                },
                {"combination_type": "comprehensive"},
                success=True
            )
            
            result = self.chain.invoke({
                "user_input": user_input.model_dump(),
                "gaps": knowledge_gap.model_dump(),
                "plan": topic_plan.model_dump(),
                "details": [detail.model_dump() for detail in topic_details]
            })
            
            # Add observation of LLM result
            explanation_builder.add_observation(
                "llm_result",
                f"LLM created combined plan with {len(result.get('recommended_resources', []))} resources and {len(result.get('success_metrics', []))} metrics",
                "llm",
                relevance_score=0.9
            )
            
            # Add validation thoughts
            explanation_builder.add_thought(
                "I need to validate and fix the data types for all fields to ensure consistency",
                confidence=0.8
            )
            
            # Ensure learning_path is a string, not a list
            if isinstance(result.get("learning_path"), list):
                result["learning_path"] = " → ".join(result["learning_path"])
                explanation_builder.add_thought("Fixed learning_path field type (converted list to string)", confidence=0.9)
            elif not isinstance(result.get("learning_path"), str):
                result["learning_path"] = "Follow the structured topics in order, practice regularly, and assess progress"
                explanation_builder.add_thought("Fixed learning_path field type with default", confidence=0.9)
            
            # Ensure other fields are the correct type
            if not isinstance(result.get("timeline"), str):
                result["timeline"] = "4-6 weeks with 2-3 hours per week"
                explanation_builder.add_thought("Fixed timeline field type", confidence=0.9)
            
            if not isinstance(result.get("recommended_resources"), list):
                result["recommended_resources"] = ["Online courses", "Practice platforms", "Community forums"]
                explanation_builder.add_thought("Fixed recommended_resources field type", confidence=0.9)
            
            if not isinstance(result.get("success_metrics"), list):
                result["success_metrics"] = ["Complete all exercises", "Pass assessments", "Apply knowledge practically"]
                explanation_builder.add_thought("Fixed success_metrics field type", confidence=0.9)
            
            # Add final thoughts
            explanation_builder.add_thought(
                f"Successfully combined all components into a cohesive plan with {len(result['recommended_resources'])} resources",
                confidence=0.8
            )
            
            # Build explanation
            cot = explanation_builder.build_chain_of_thought(
                "Comprehensive plan combination using LLM",
                f"Combined gap analysis, topic plan, and {len(topic_details)} topic details into a cohesive learning plan with clear progression and success metrics.",
                f"Created combined plan with {len(result['recommended_resources'])} resources and {len(result['success_metrics'])} metrics",
                confidence=0.8
            )
            
            explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.8)
            
            # Set output summary
            explanation_builder.set_output_summary(result)
            
            return result, explanation
            
        except Exception as e:
            # Add action for error case
            explanation_builder.add_action(
                "error_handling",
                f"Handling error in plan combination: {str(e)}",
                {"error": str(e)},
                {"error_handled": True},
                success=False,
                error_message=str(e)
            )
            
            # Add observation of error
            explanation_builder.add_observation(
                "error_occurred",
                f"Error occurred during LLM combination: {str(e)}",
                "system",
                relevance_score=1.0
            )
            
            print(f"Error in plan combination: {e}")
            
            # Build explanation for error case
            cot = explanation_builder.build_chain_of_thought(
                "Error handling in plan combination",
                f"Encountered error during LLM combination: {str(e)}. Falling back to default combination.",
                "Use default combined plan due to error",
                confidence=0.4
            )
            
            explanation = explanation_builder.build_explanation(chain_of_thought=cot, confidence_score=0.4)
            
            # Set output summary for error case
            explanation_builder.set_output_summary({
                "learning_path": "Follow the structured topics in order, practice regularly, and assess progress",
                "recommended_resources": ["Online courses", "Practice platforms", "Community forums"],
                "timeline": "4-6 weeks with 2-3 hours per week",
                "success_metrics": ["Complete all exercises", "Pass assessments", "Apply knowledge practically"]
            })
            
            return {
                "learning_path": "Follow the structured topics in order, practice regularly, and assess progress",
                "recommended_resources": ["Online courses", "Practice platforms", "Community forums"],
                "timeline": "4-6 weeks with 2-3 hours per week",
                "success_metrics": ["Complete all exercises", "Pass assessments", "Apply knowledge practically"]
            }, explanation 