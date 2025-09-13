"""
Learning Plan Generator using LangChain AI 2 and LangGraph
A three-agent system that creates personalized learning plans
"""

import os
import json
import numpy as np
from typing import Dict, Any

from models import UserInput, LearningFormat
from workflow import LearningPlanWorkflow
from config import GOOGLE_API_KEY
from explanation_display import ExplanationDisplay, display_explanations_interactive
from integrated_explainer import IntegratedExplainabilitySystem
from explanation_visualizer import AdvancedExplanationVisualizer

def print_learning_plan(plan: Dict[str, Any]) -> None:
    """Pretty print the complete learning plan"""
    print("\n" + "="*80)
    print("🎓 COMPLETE LEARNING PLAN")
    print("="*80)
    
    # User Input Summary
    print(f"\n📋 TOPIC: {plan['user_input']['topic']}")
    print(f"🎯 PREFERRED FORMAT: {plan['user_input']['preferred_format']}")
    print(f"📚 BACKGROUND: {plan['user_input']['background'][:100]}...")
    
    # Knowledge Gap Analysis
    print(f"\n🔍 KNOWLEDGE GAP ANALYSIS")
    print("-" * 40)
    gap = plan['knowledge_gap']
    print(f"Current Level: {gap['current_level']}")
    print(f"Target Level: {gap['target_level']}")
    print(f"Identified Gaps: {', '.join(gap['identified_gaps'])}")
    print(f"Analysis: {gap['gap_analysis']}")
    
    # Topic Plan
    print(f"\n📚 TOPIC PLAN")
    print("-" * 40)
    topic_plan = plan['topic_plan']
    print(f"Main Topics: {', '.join(topic_plan['main_topics'])}")
    print(f"Learning Objectives: {', '.join(topic_plan['learning_objectives'])}")
    print(f"Estimated Duration: {topic_plan['estimated_duration']}")
    
    # Topic Details
    print(f"\n📝 DETAILED TOPIC BREAKDOWN")
    print("-" * 40)
    for i, detail in enumerate(plan['topic_details'], 1):
        print(f"\n{i}. {detail['topic_name']}")
        print(f"   Description: {detail['description']}")
        print(f"   Resources: {', '.join(detail['resources'])}")
        print(f"   Exercises: {', '.join(detail['exercises'])}")
        print(f"   Assessment: {detail['assessment_criteria']}")
    
    # Combined Plan
    print(f"\n🔗 COMPLETE LEARNING PATH")
    print("-" * 40)
    print(f"Learning Path: {plan['learning_path']}")
    print(f"Timeline: {plan['timeline']}")
    print(f"Success Metrics: {', '.join(plan['success_metrics'])}")
    print(f"Overall Resources: {', '.join(plan['recommended_resources'])}")
    
    print("\n" + "="*80)

def get_user_input() -> UserInput:
    """Get user input for creating a learning plan"""
    print("🎓 LEARNING PLAN GENERATOR")
    print("=" * 50)
    
    # Get topic
    topic = input("\n📚 What topic would you like to learn today? ").strip()
    while not topic:
        print("❌ Topic cannot be empty. Please try again.")
        topic = input("📚 What topic would you like to learn today? ").strip()
    
    # Get background
    print("\n📖 Please describe your current background and knowledge level:")
    print("   (e.g., 'I'm a beginner with no programming experience' or 'I have intermediate Python skills')")
    background = input("Your background: ").strip()
    while not background:
        print("❌ Background cannot be empty. Please try again.")
        background = input("Your background: ").strip()
    
    # Get preferred format
    print("\n🎯 Choose your preferred learning format:")
    print("   1. Video")
    print("   2. Text")
    print("   3. Audio")
    
    while True:
        try:
            choice = int(input("Enter your choice (1-3): "))
            if choice == 1:
                preferred_format = LearningFormat.VIDEO
                break
            elif choice == 2:
                preferred_format = LearningFormat.TEXT
                break
            elif choice == 3:
                preferred_format = LearningFormat.AUDIO
                break
            else:
                print("❌ Please enter a number between 1 and 3.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Single iteration workflow - no need for user input
    max_iterations = 1
    
    return UserInput(
        topic=topic,
        background=background,
        preferred_format=preferred_format,
        max_iterations=max_iterations
    )

def main():
    """Main function to run the learning plan generator"""
    
    # Check for API key
    if not GOOGLE_API_KEY:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
        return
    
    try:
        # Get user input
        user_input = get_user_input()
        
        # Create and run the workflow
        print(f"\n🚀 Creating your personalized learning plan...")
        print(f"   This may take a few moments as we analyze your needs...")
        
        workflow = LearningPlanWorkflow()
        learning_plan = workflow.create_learning_plan(user_input)
        
        # Display the complete plan
        print_learning_plan(learning_plan.model_dump())
        
        # Display explanations
        explanations = workflow.get_explanations()
        if explanations:
            print(f"\n🧠 AGENT EXPLANATIONS")
            print(f"Total agent decisions: {len(explanations)}")
            
            explanation_choice = input("\n🔍 Would you like to view agent explanations? (y/n): ").lower().strip()
            if explanation_choice in ['y', 'yes']:
                display_explanations_interactive(explanations)
            
            # Advanced explainability with LIME/SHAP
            advanced_choice = input("\n🔬 Would you like advanced model-agnostic explanations (LIME/SHAP)? (y/n): ").lower().strip()
            if advanced_choice in ['y', 'yes']:
                try:
                    # Check if feature extractor and predictor are available
                    if not hasattr(workflow, 'feature_extractor') or workflow.feature_extractor is None:
                        print("⚠️ Feature extractor not available. Creating new one...")
                        from feature_extractor import LearningPlanFeatureExtractor
                        workflow.feature_extractor = LearningPlanFeatureExtractor()
                    
                    if not hasattr(workflow, 'predictor') or workflow.predictor is None:
                        print("⚠️ Predictor not available. Creating new one...")
                        from feature_extractor import LearningPlanPredictor
                        workflow.predictor = LearningPlanPredictor()
                        
                        # Fit predictor with sample data
                        from feature_extractor import create_sample_learning_plans
                        sample_plans = create_sample_learning_plans()
                        sample_plans.append(learning_plan)
                        X, feature_names = workflow.feature_extractor.fit_transform(sample_plans)
                        workflow.feature_extractor.feature_names = feature_names
                        y = np.array([0.8, 0.9, 0.7])
                        workflow.predictor.fit(X, y)
                        print("✅ Predictor fitted with sample data")
                    
                    # Create integrated explainability system
                    integrated_system = IntegratedExplainabilitySystem(
                        workflow.feature_extractor,
                        workflow.predictor,
                        workflow.explanation_logger
                    )
                    
                    # Generate integrated explanation
                    integrated_explanation = integrated_system.explain_learning_plan(
                        learning_plan, explanations, include_lime=True, include_shap=True
                    )
                    
                    # Display integrated explanation
                    print("\n" + "="*80)
                    print("🔬 INTEGRATED EXPLANATION (LIME + SHAP + Agent Reasoning)")
                    print("="*80)
                    integrated_system.print_explanation(integrated_explanation)
                    
                    # Visualization option
                    viz_choice = input("\n📊 Would you like to see visualizations? (y/n): ").lower().strip()
                    if viz_choice in ['y', 'yes']:
                        visualizer = AdvancedExplanationVisualizer()
                        visualizer.plot_integrated_explanation(integrated_explanation)
                    
                    # Save integrated explanation
                    save_integrated_choice = input("\n💾 Save integrated explanation? (y/n): ").lower().strip()
                    if save_integrated_choice in ['y', 'yes']:
                        integrated_file = f"integrated_explanation_{user_input.topic.replace(' ', '_').lower()}.json"
                        integrated_system.save_explanation(integrated_explanation, integrated_file)
                        print(f"✅ Integrated explanation saved to {integrated_file}")
                        
                except Exception as e:
                    print(f"⚠️ Could not generate advanced explanations: {e}")
                    print("   Falling back to basic agent explanations only.")
            
            # Save explanations option
            save_explanations_choice = input("\n💾 Would you like to save explanations to a file? (y/n): ").lower().strip()
            if save_explanations_choice in ['y', 'yes']:
                explanation_file = workflow.save_explanations()
                print(f"✅ Explanations saved to {explanation_file}")
                
                # Also save audit trail
                audit_trail = workflow.get_audit_trail()
                audit_file = f"audit_trail_{user_input.topic.replace(' ', '_').lower()}.json"
                with open(audit_file, 'w') as f:
                    json.dump(audit_trail, f, indent=2, default=str)
                print(f"✅ Audit trail saved to {audit_file}")
        
        # Save to file option
        save_choice = input("\n💾 Would you like to save this plan to a file? (y/n): ").lower().strip()
        if save_choice in ['y', 'yes']:
            filename = f"learning_plan_{user_input.topic.replace(' ', '_').lower()}.json"
            with open(filename, 'w') as f:
                json.dump(learning_plan.model_dump(), f, indent=2)
            print(f"✅ Plan saved to {filename}")
        
        print("\n🎉 Thank you for using the Learning Plan Generator!")
        print("   Good luck with your learning journey!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    main() 