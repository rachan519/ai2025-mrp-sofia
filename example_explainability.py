#!/usr/bin/env python3
"""
Example script demonstrating the explainability features of the learning plan generator.
This script shows how to access and display agent explanations.
"""

from models import UserInput, LearningFormat
from workflow import LearningPlanWorkflow
from explanation_display import ExplanationDisplay, display_explanations_interactive
from config import GOOGLE_API_KEY


def demonstrate_explainability():
    """Demonstrate the explainability features"""
    
    if not GOOGLE_API_KEY:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please set your Gemini API key in the .env file or as an environment variable.")
        return
    
    # Create a sample user input
    user_input = UserInput(
        topic="Machine Learning",
        background="I have basic programming experience in Python but no knowledge of machine learning concepts",
        preferred_format=LearningFormat.VIDEO,
        max_iterations=1
    )
    
    print("🎓 LEARNING PLAN GENERATOR - EXPLAINABILITY DEMO")
    print("="*60)
    print(f"Topic: {user_input.topic}")
    print(f"Background: {user_input.background}")
    print(f"Preferred Format: {user_input.preferred_format.value}")
    print("="*60)
    
    try:
        # Create workflow
        workflow = LearningPlanWorkflow()
        
        # Generate learning plan
        print("\n🚀 Generating learning plan with explainability...")
        learning_plan = workflow.create_learning_plan(user_input)
        
        # Get explanations
        explanations = workflow.get_explanations()
        
        print(f"\n✅ Learning plan generated successfully!")
        print(f"📊 Total agent decisions: {len(explanations)}")
        
        # Display explanation summary
        print("\n" + "="*60)
        print("🧠 AGENT EXPLANATIONS SUMMARY")
        print("="*60)
        ExplanationDisplay.print_explanation_summary(explanations)
        
        # Display detailed explanations
        print("\n" + "="*60)
        print("🔍 DETAILED AGENT EXPLANATIONS")
        print("="*60)
        ExplanationDisplay.print_all_explanations(explanations)
        
        # Display audit trail
        print("\n" + "="*60)
        print("📊 AUDIT TRAIL")
        print("="*60)
        audit_trail = workflow.get_audit_trail()
        ExplanationDisplay.print_audit_trail(audit_trail)
        
        # Save explanations
        print("\n💾 Saving explanations...")
        explanation_file = workflow.save_explanations()
        print(f"✅ Explanations saved to: {explanation_file}")
        
        # Interactive explanation viewer
        print("\n🎯 INTERACTIVE EXPLANATION VIEWER")
        print("="*60)
        display_explanations_interactive(explanations)
        
        print("\n🎉 Explainability demonstration completed!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_chain_of_thought():
    """Demonstrate Chain-of-Thought reasoning specifically"""
    
    if not GOOGLE_API_KEY:
        print("❌ Error: GOOGLE_API_KEY not found.")
        return
    
    print("\n🧠 CHAIN-OF-THOUGHT DEMONSTRATION")
    print("="*50)
    
    user_input = UserInput(
        topic="Python Programming",
        background="Complete beginner with no programming experience",
        preferred_format=LearningFormat.TEXT,
        max_iterations=1
    )
    
    workflow = LearningPlanWorkflow()
    
    try:
        # Get just the gap analysis to see CoT reasoning
        gap_agent = workflow.gap_agent
        knowledge_gap, explanation = gap_agent.analyze_gaps(user_input)
        
        print(f"Agent: {explanation.agent_name}")
        print(f"Task: {explanation.task_description}")
        print(f"Confidence: {explanation.confidence_score:.2f}")
        
        if explanation.chain_of_thought:
            print(f"\nChain-of-Thought Reasoning:")
            print(f"Context: {explanation.chain_of_thought.decision_context}")
            print(f"Decision: {explanation.chain_of_thought.decision}")
            
            print(f"\nThought Steps:")
            for step in explanation.chain_of_thought.thought_steps:
                print(f"  Step {step.step_number}: {step.thought}")
                print(f"    └─ Confidence: {step.confidence:.2f}")
            
            print(f"\nFinal Reasoning: {explanation.chain_of_thought.final_reasoning}")
        
    except Exception as e:
        print(f"❌ Error in CoT demonstration: {e}")


def demonstrate_react_pattern():
    """Demonstrate ReAct pattern specifically"""
    
    if not GOOGLE_API_KEY:
        print("❌ Error: GOOGLE_API_KEY not found.")
        return
    
    print("\n🔄 REACT PATTERN DEMONSTRATION")
    print("="*50)
    
    user_input = UserInput(
        topic="Data Science",
        background="I know Python basics and some statistics",
        preferred_format=LearningFormat.VIDEO,
        max_iterations=1
    )
    
    workflow = LearningPlanWorkflow()
    
    try:
        # Get topic planning to see ReAct pattern
        gap_agent = workflow.gap_agent
        planning_agent = workflow.planning_agent
        
        knowledge_gap, gap_explanation = gap_agent.analyze_gaps(user_input)
        topic_plan, plan_explanation = planning_agent.create_plan(user_input, knowledge_gap)
        
        print(f"Agent: {plan_explanation.agent_name}")
        print(f"Task: {plan_explanation.task_description}")
        
        if plan_explanation.react_trace:
            print(f"\nReAct Trace:")
            print(f"Task Context: {plan_explanation.react_trace.task_context}")
            print(f"Final Outcome: {plan_explanation.react_trace.final_outcome}")
            print(f"Success: {'✅' if plan_explanation.react_trace.success else '❌'}")
            
            print(f"\nStep-by-Step Process:")
            for i, step in enumerate(plan_explanation.react_trace.steps, 1):
                if hasattr(step, 'thought'):
                    print(f"  {i}. THOUGHT: {step.thought}")
                elif hasattr(step, 'action_description'):
                    print(f"  {i}. ACTION: {step.action_description}")
                    print(f"      └─ Type: {step.action_type}, Success: {'✅' if step.success else '❌'}")
                elif hasattr(step, 'observation_content'):
                    print(f"  {i}. OBSERVATION: {step.observation_content}")
                    print(f"      └─ Source: {step.source}")
        
    except Exception as e:
        print(f"❌ Error in ReAct demonstration: {e}")


if __name__ == "__main__":
    print("🎯 EXPLAINABILITY FEATURES DEMONSTRATION")
    print("="*60)
    
    # Run demonstrations
    demonstrate_explainability()
    demonstrate_chain_of_thought()
    demonstrate_react_pattern()
    
    print("\n🎉 All demonstrations completed!")
