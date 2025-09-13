#!/usr/bin/env python3
"""
Example usage of the Learning Plan Generator
Demonstrates how to use the system programmatically
"""

from models import UserInput, LearningFormat
from workflow import LearningPlanWorkflow

def example_usage():
    """Example of how to use the learning plan generator programmatically"""
    
    # Example 1: Learning Python Programming
    print("🐍 Example 1: Learning Python Programming")
    print("-" * 50)
    
    python_input = UserInput(
        topic="Python Programming",
        background="I'm a complete beginner with no programming experience. I've never written code before.",
        preferred_format=LearningFormat.VIDEO,
        max_iterations=1
    )
    
    try:
        workflow = LearningPlanWorkflow()
        python_plan = workflow.create_learning_plan(python_input)
        
        print(f"✅ Plan created successfully!")
        print(f"📚 Main topics: {len(python_plan.topic_plan.main_topics)}")
        print(f"📝 Detailed topics: {len(python_plan.topic_details)}")
        print(f"⏱️  Estimated duration: {python_plan.topic_plan.estimated_duration}")
        
    except Exception as e:
        print(f"❌ Error creating Python plan: {e}")
    
    print("\n" + "="*80)
    
    # Example 2: Learning Machine Learning
    print("🤖 Example 2: Learning Machine Learning")
    print("-" * 50)
    
    ml_input = UserInput(
        topic="Machine Learning",
        background="I have intermediate Python skills and basic understanding of statistics. I've taken a few data science courses.",
        preferred_format=LearningFormat.TEXT,
        max_iterations=1
    )
    
    try:
        workflow = LearningPlanWorkflow()
        ml_plan = workflow.create_learning_plan(ml_input)
        
        print(f"✅ Plan created successfully!")
        print(f"📚 Main topics: {len(ml_plan.topic_plan.main_topics)}")
        print(f"📝 Detailed topics: {len(ml_plan.topic_details)}")
        print(f"⏱️  Estimated duration: {ml_plan.topic_plan.estimated_duration}")
        
    except Exception as e:
        print(f"❌ Error creating ML plan: {e}")
    
    print("\n" + "="*80)
    
    # Example 3: Learning Spanish
    print("🇪🇸 Example 3: Learning Spanish")
    print("-" * 50)
    
    spanish_input = UserInput(
        topic="Spanish Language",
        background="I know a few basic phrases and can count to 10. I've never taken formal Spanish classes.",
        preferred_format=LearningFormat.AUDIO,
        max_iterations=1
    )
    
    try:
        workflow = LearningPlanWorkflow()
        spanish_plan = workflow.create_learning_plan(spanish_input)
        
        print(f"✅ Plan created successfully!")
        print(f"📚 Main topics: {len(spanish_plan.topic_plan.main_topics)}")
        print(f"📝 Detailed topics: {len(spanish_plan.topic_details)}")
        print(f"⏱️  Estimated duration: {spanish_plan.topic_plan.estimated_duration}")
        
    except Exception as e:
        print(f"❌ Error creating Spanish plan: {e}")

def custom_plan_example():
    """Example of creating a custom learning plan"""
    
    print("\n🎯 Custom Learning Plan Example")
    print("-" * 40)
    
    # Get custom input
    topic = input("Enter a topic you want to learn: ").strip()
    if not topic:
        topic = "Web Development"
        print(f"Using default topic: {topic}")
    
    background = input("Describe your background (or press Enter for default): ").strip()
    if not background:
        background = "I have basic HTML knowledge and want to learn modern web development."
        print(f"Using default background: {background}")
    
    print("Choose format: 1-Video, 2-Text, 3-Audio")
    format_choice = input("Format choice (1-3, or Enter for default): ").strip()
    
    if format_choice == "1":
        preferred_format = LearningFormat.VIDEO
    elif format_choice == "2":
        preferred_format = LearningFormat.TEXT
    elif format_choice == "3":
        preferred_format = LearningFormat.AUDIO
    else:
        preferred_format = LearningFormat.VIDEO
        print("Using default format: Video")
    
    # Single iteration workflow
    max_iterations = 1
    
    custom_input = UserInput(
        topic=topic,
        background=background,
        preferred_format=preferred_format,
        max_iterations=max_iterations
    )
    
    try:
        workflow = LearningPlanWorkflow()
        custom_plan = workflow.create_learning_plan(custom_input)
        
        print(f"\n✅ Custom plan created successfully!")
        print(f"📋 Topic: {custom_plan.user_input.topic}")
        print(f"🎯 Format: {custom_plan.user_input.preferred_format.value}")
        print(f"📚 Main topics: {', '.join(custom_plan.topic_plan.main_topics)}")
        print(f"⏱️  Duration: {custom_plan.topic_plan.estimated_duration}")
        
        # Show first topic detail as example
        if custom_plan.topic_details:
            first_topic = custom_plan.topic_details[0]
            print(f"\n📝 Example topic detail:")
            print(f"   Topic: {first_topic.topic_name}")
            print(f"   Description: {first_topic.description[:100]}...")
            print(f"   Resources: {', '.join(first_topic.resources[:2])}")
        
    except Exception as e:
        print(f"❌ Error creating custom plan: {e}")

if __name__ == "__main__":
    print("🚀 Learning Plan Generator - Example Usage")
    print("=" * 60)
    
    print("\nChoose an example to run:")
    print("1. Pre-defined examples (Python, ML, Spanish)")
    print("2. Custom learning plan")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            example_usage()
            break
        elif choice == "2":
            custom_plan_example()
            break
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Please enter 1, 2, or 3.") 