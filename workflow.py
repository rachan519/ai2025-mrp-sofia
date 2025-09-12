from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
# MemorySaver not needed for basic workflow
# ToolExecutor not needed for this implementation
import json

from models import AgentState, CompleteLearningPlan, UserInput, AgentExplanation
from agents import GapAnalysisAgent, TopicPlanningAgent, TopicDetailAgent, PlanCombinerAgent
from explanation_logger import ExplanationLogger

class LearningPlanWorkflow:
    """LangGraph workflow for creating personalized learning plans"""
    
    def __init__(self):
        # Single iteration workflow - no need for max_iterations
        self.gap_agent = GapAnalysisAgent()
        self.planning_agent = TopicPlanningAgent()
        self.detail_agent = TopicDetailAgent()
        self.combiner_agent = PlanCombinerAgent()
        
        # Initialize explanation logger
        self.explanation_logger = ExplanationLogger()
        
        # Memory saver not needed for basic workflow
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with three agents"""
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("gap_analysis", self._gap_analysis_node)
        workflow.add_node("topic_planning", self._topic_planning_node)
        workflow.add_node("topic_detailing", self._topic_detailing_node)
        workflow.add_node("plan_combination", self._plan_combination_node)
        
        # Define the workflow edges - simple linear flow
        workflow.set_entry_point("gap_analysis")
        workflow.add_edge("gap_analysis", "topic_planning")
        workflow.add_edge("topic_planning", "topic_detailing")
        workflow.add_edge("topic_detailing", "plan_combination")
        workflow.add_edge("plan_combination", END)
        
        return workflow.compile()
    
    def _gap_analysis_node(self, state: AgentState) -> AgentState:
        """Execute gap analysis agent"""
        print(f"\n🔍 Analyzing knowledge gaps...")
        
        # Analyze gaps using the first agent (now returns tuple with explanation)
        knowledge_gap, explanation = self.gap_agent.analyze_gaps(state.user_input)
        
        # Log the explanation
        self.explanation_logger.log_explanation(explanation)
        
        # Update state
        state.knowledge_gap = knowledge_gap
        state.current_step = "gap_analysis"
        state.explanations.append(explanation)
        
        print(f"✅ Knowledge gaps identified: {len(knowledge_gap.identified_gaps)} gaps found")
        print(f"🧠 Agent reasoning: {explanation.chain_of_thought.final_reasoning if explanation.chain_of_thought else 'N/A'}")
        return state
    
    def _topic_planning_node(self, state: AgentState) -> AgentState:
        """Execute topic planning agent"""
        print(f"📚 Creating topic plan...")
        
        # Create topic plan using the second agent (now returns tuple with explanation)
        topic_plan, explanation = self.planning_agent.create_plan(state.user_input, state.knowledge_gap)
        
        # Log the explanation
        self.explanation_logger.log_explanation(explanation)
        
        # Update state
        state.topic_plan = topic_plan
        state.current_step = "topic_planning"
        state.explanations.append(explanation)
        
        print(f"✅ Topic plan created: {len(topic_plan.main_topics)} main topics")
        print(f"🧠 Agent reasoning: {explanation.chain_of_thought.final_reasoning if explanation.chain_of_thought else 'N/A'}")
        return state
    
    def _topic_detailing_node(self, state: AgentState) -> AgentState:
        """Execute topic detail agent for each topic"""
        print(f"📝 Creating topic details...")
        
        topic_details = []
        explanations = []
        
        # Create detailed breakdown for each main topic
        for i, main_topic in enumerate(state.topic_plan.main_topics):
            objective = state.topic_plan.learning_objectives[i] if i < len(state.topic_plan.learning_objectives) else "Learn the topic"
            
            # Create topic detail (now returns tuple with explanation)
            detail, explanation = self.detail_agent.create_topic_detail(
                main_topic, main_topic, objective, state.user_input
            )
            topic_details.append(detail)
            explanations.append(explanation)
            
            # Log the explanation
            self.explanation_logger.log_explanation(explanation)
        
        # Update state
        state.topic_details = topic_details
        state.current_step = "topic_detailing"
        state.explanations.extend(explanations)
        
        print(f"✅ Topic details created: {len(topic_details)} detailed topics")
        print(f"🧠 Agent reasoning: {explanations[0].chain_of_thought.final_reasoning if explanations and explanations[0].chain_of_thought else 'N/A'}")
        return state
    
    def _plan_combination_node(self, state: AgentState) -> AgentState:
        """Execute plan combination agent"""
        print(f"🔗 Combining learning plan...")
        
        # Combine all components into a complete plan (now returns tuple with explanation)
        combined_plan, explanation = self.combiner_agent.combine_plan(
            state.user_input, state.knowledge_gap, state.topic_plan, state.topic_details
        )
        
        # Log the explanation
        self.explanation_logger.log_explanation(explanation)
        
        # Create the complete learning plan with validation
        try:
            complete_plan = CompleteLearningPlan(
                user_input=state.user_input,
                knowledge_gap=state.knowledge_gap,
                topic_plan=state.topic_plan,
                topic_details=state.topic_details,
                **combined_plan
            )
        except Exception as validation_error:
            print(f"⚠️ Validation error in plan creation: {validation_error}")
            # Create a fallback plan with safe defaults
            complete_plan = CompleteLearningPlan(
                user_input=state.user_input,
                knowledge_gap=state.knowledge_gap,
                topic_plan=state.topic_plan,
                topic_details=state.topic_details,
                learning_path="Follow the structured topics in order, practice regularly, and assess progress",
                recommended_resources=["Online courses", "Practice platforms", "Community forums"],
                timeline="4-6 weeks with 2-3 hours per week",
                success_metrics=["Complete all exercises", "Pass assessments", "Apply knowledge practically"]
            )
        
        # Update state
        state.complete_plan = complete_plan
        state.current_step = "plan_combination"
        state.explanations.append(explanation)
        
        print(f"✅ Complete learning plan created successfully!")
        print(f"🧠 Agent reasoning: {explanation.chain_of_thought.final_reasoning if explanation.chain_of_thought else 'N/A'}")
        return state
    
    # Iteration logic removed - single pass workflow
    
    def create_learning_plan(self, user_input: UserInput) -> CompleteLearningPlan:
        """Execute the complete workflow to create a learning plan"""
        
        # Initialize the agent state
        initial_state = AgentState(
            user_input=user_input,
            max_iterations=1  # Single iteration workflow
        )
        
        print(f"🚀 Starting learning plan creation workflow...")
        print(f"📋 Topic: {user_input.topic}")
        print(f"🎯 Preferred format: {user_input.preferred_format.value}")
        print(f"📚 Background: {user_input.background[:100]}...")
        
        # Execute the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            # LangGraph returns a dictionary, so we need to access it differently
            if isinstance(final_state, dict) and 'complete_plan' in final_state:
                print(f"\n🎉 Learning plan creation completed successfully!")
                return final_state['complete_plan']
            elif hasattr(final_state, 'complete_plan') and final_state.complete_plan:
                print(f"\n🎉 Learning plan creation completed successfully!")
                return final_state.complete_plan
            else:
                raise Exception("Workflow completed but no learning plan was generated")
                
        except Exception as e:
            print(f"❌ Error in workflow execution: {e}")
            raise
    
    def get_workflow_status(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get the current status of the workflow"""
        return {
            "max_iterations": self.max_iterations,
            "current_iteration": config.get("iteration_count", 0),
            "current_step": config.get("current_step", "not_started"),
            "workflow_complete": config.get("workflow_complete", False)
        }
    
    def get_explanations(self) -> List[AgentExplanation]:
        """Get all explanations from the current session"""
        return self.explanation_logger.get_session_explanations()
    
    def get_explanations_for_agent(self, agent_name: str) -> List[AgentExplanation]:
        """Get explanations for a specific agent"""
        return self.explanation_logger.get_explanations_for_agent(agent_name)
    
    def save_explanations(self) -> str:
        """Save all explanations to a file and return the file path"""
        return self.explanation_logger.save_session_summary()
    
    def get_audit_trail(self) -> Dict[str, Any]:
        """Get a comprehensive audit trail of all agent decisions"""
        return self.explanation_logger.create_audit_trail() 