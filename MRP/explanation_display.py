"""
Explanation Display Module
Provides functionality to display agent explanations in a user-friendly format.
"""

from typing import List, Dict, Any
from models import AgentExplanation, ChainOfThought, ReActTrace, ThoughtStep, ActionStep, ObservationStep
from datetime import datetime


class ExplanationDisplay:
    """Display agent explanations in various formats"""
    
    @staticmethod
    def print_agent_explanation(explanation: AgentExplanation) -> None:
        """Print a single agent explanation in a formatted way"""
        print(f"\n{'='*80}")
        print(f"🤖 AGENT: {explanation.agent_name}")
        print(f"📋 TASK: {explanation.task_description}")
        print(f"⏱️  PROCESSING TIME: {explanation.processing_time:.2f} seconds")
        print(f"🎯 CONFIDENCE: {explanation.confidence_score:.2f}")
        print(f"📅 TIMESTAMP: {explanation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Print Chain-of-Thought reasoning
        if explanation.chain_of_thought:
            ExplanationDisplay._print_chain_of_thought(explanation.chain_of_thought)
        
        # Print ReAct trace
        if explanation.react_trace:
            ExplanationDisplay._print_react_trace(explanation.react_trace)
        
        # Print decision factors
        if explanation.decision_factors:
            print(f"\n🔍 DECISION FACTORS:")
            for i, factor in enumerate(explanation.decision_factors, 1):
                print(f"   {i}. {factor}")
        
        # Print alternative approaches
        if explanation.alternative_approaches:
            print(f"\n🔄 ALTERNATIVE APPROACHES CONSIDERED:")
            for i, approach in enumerate(explanation.alternative_approaches, 1):
                print(f"   {i}. {approach}")
        
        # Print limitations
        if explanation.limitations:
            print(f"\n⚠️  KNOWN LIMITATIONS:")
            for i, limitation in enumerate(explanation.limitations, 1):
                print(f"   {i}. {limitation}")
        
        print(f"\n{'='*80}")
    
    @staticmethod
    def _print_chain_of_thought(cot: ChainOfThought) -> None:
        """Print Chain-of-Thought reasoning"""
        print(f"\n🧠 CHAIN-OF-THOUGHT REASONING:")
        print(f"   Context: {cot.decision_context}")
        print(f"   Final Decision: {cot.decision}")
        print(f"   Overall Confidence: {cot.confidence:.2f}")
        
        print(f"\n   Thought Process:")
        for step in cot.thought_steps:
            print(f"   Step {step.step_number}: {step.thought}")
            print(f"   └─ Confidence: {step.confidence:.2f}")
        
        print(f"\n   Final Reasoning: {cot.final_reasoning}")
    
    @staticmethod
    def _print_react_trace(react: ReActTrace) -> None:
        """Print ReAct trace"""
        print(f"\n🔄 REACT TRACE:")
        print(f"   Task Context: {react.task_context}")
        print(f"   Final Outcome: {react.final_outcome}")
        print(f"   Success: {'✅' if react.success else '❌'}")
        print(f"   Total Steps: {react.total_steps}")
        
        print(f"\n   Step-by-Step Process:")
        for i, step in enumerate(react.steps, 1):
            if isinstance(step, ThoughtStep):
                print(f"   {i}. THOUGHT: {step.thought}")
                print(f"      └─ Confidence: {step.confidence:.2f}")
            elif isinstance(step, ActionStep):
                print(f"   {i}. ACTION: {step.action_description}")
                print(f"      └─ Type: {step.action_type}, Success: {'✅' if step.success else '❌'}")
                if step.error_message:
                    print(f"      └─ Error: {step.error_message}")
            elif isinstance(step, ObservationStep):
                print(f"   {i}. OBSERVATION: {step.observation_content}")
                print(f"      └─ Source: {step.source}, Relevance: {step.relevance_score:.2f}")
    
    @staticmethod
    def print_all_explanations(explanations: List[AgentExplanation]) -> None:
        """Print all explanations in sequence"""
        print(f"\n🎯 AGENT EXPLANATIONS SUMMARY")
        print(f"Total Agents: {len(set(exp.agent_name for exp in explanations))}")
        print(f"Total Decisions: {len(explanations)}")
        
        for explanation in explanations:
            ExplanationDisplay.print_agent_explanation(explanation)
    
    @staticmethod
    def print_explanations_by_agent(explanations: List[AgentExplanation]) -> None:
        """Print explanations grouped by agent"""
        agent_groups = {}
        for exp in explanations:
            if exp.agent_name not in agent_groups:
                agent_groups[exp.agent_name] = []
            agent_groups[exp.agent_name].append(exp)
        
        for agent_name, agent_explanations in agent_groups.items():
            print(f"\n🤖 {agent_name.upper()} EXPLANATIONS")
            print(f"Number of decisions: {len(agent_explanations)}")
            print(f"{'='*60}")
            
            for i, explanation in enumerate(agent_explanations, 1):
                print(f"\nDecision {i}:")
                print(f"  Task: {explanation.task_description}")
                print(f"  Confidence: {explanation.confidence_score:.2f}")
                print(f"  Processing Time: {explanation.processing_time:.2f}s")
                
                if explanation.chain_of_thought:
                    print(f"  Reasoning: {explanation.chain_of_thought.final_reasoning}")
    
    @staticmethod
    def print_audit_trail(audit_trail: Dict[str, Any]) -> None:
        """Print audit trail in a formatted way"""
        print(f"\n📊 AUDIT TRAIL")
        print(f"{'='*60}")
        print(f"Session ID: {audit_trail['session_id']}")
        print(f"Created At: {audit_trail['created_at']}")
        print(f"Total Agents: {audit_trail['total_agents']}")
        print(f"Total Decisions: {audit_trail['total_decisions']}")
        
        print(f"\n🤖 AGENT SUMMARIES:")
        for agent_name, summary in audit_trail['agent_summaries'].items():
            print(f"\n  {agent_name}:")
            print(f"    Decisions: {summary['total_decisions']}")
            print(f"    Avg Confidence: {summary['average_confidence']}")
            print(f"    Avg Processing Time: {summary['average_processing_time']}s")
            print(f"    Key Decision Factors: {', '.join(summary['decision_factors'][:3])}...")
            print(f"    Key Limitations: {', '.join(summary['limitations'][:2])}...")
    
    @staticmethod
    def print_explanation_summary(explanations: List[AgentExplanation]) -> None:
        """Print a summary of all explanations"""
        if not explanations:
            print("No explanations available.")
            return
        
        print(f"\n📋 EXPLANATION SUMMARY")
        print(f"{'='*50}")
        
        # Group by agent
        agent_stats = {}
        for exp in explanations:
            if exp.agent_name not in agent_stats:
                agent_stats[exp.agent_name] = {
                    'count': 0,
                    'total_confidence': 0,
                    'total_time': 0,
                    'tasks': []
                }
            
            agent_stats[exp.agent_name]['count'] += 1
            agent_stats[exp.agent_name]['total_confidence'] += exp.confidence_score
            agent_stats[exp.agent_name]['total_time'] += exp.processing_time
            agent_stats[exp.agent_name]['tasks'].append(exp.task_description)
        
        # Print statistics
        for agent_name, stats in agent_stats.items():
            avg_confidence = stats['total_confidence'] / stats['count']
            avg_time = stats['total_time'] / stats['count']
            
            print(f"\n🤖 {agent_name}:")
            print(f"   Decisions: {stats['count']}")
            print(f"   Avg Confidence: {avg_confidence:.2f}")
            print(f"   Avg Processing Time: {avg_time:.2f}s")
            print(f"   Tasks: {', '.join(stats['tasks'][:2])}...")
        
        # Overall statistics
        total_confidence = sum(exp.confidence_score for exp in explanations)
        total_time = sum(exp.processing_time for exp in explanations)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Decisions: {len(explanations)}")
        print(f"   Average Confidence: {total_confidence / len(explanations):.2f}")
        print(f"   Total Processing Time: {total_time:.2f}s")
        print(f"   Average Processing Time: {total_time / len(explanations):.2f}s")


def display_explanations_interactive(explanations: List[AgentExplanation]) -> None:
    """Interactive explanation display with user choices"""
    if not explanations:
        print("No explanations available.")
        return
    
    while True:
        print(f"\n🎯 EXPLANATION VIEWER")
        print(f"{'='*40}")
        print(f"1. View all explanations")
        print(f"2. View by agent")
        print(f"3. View summary")
        print(f"4. View specific explanation")
        print(f"5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            ExplanationDisplay.print_all_explanations(explanations)
        elif choice == '2':
            ExplanationDisplay.print_explanations_by_agent(explanations)
        elif choice == '3':
            ExplanationDisplay.print_explanation_summary(explanations)
        elif choice == '4':
            print(f"\nAvailable explanations:")
            for i, exp in enumerate(explanations, 1):
                print(f"  {i}. {exp.agent_name} - {exp.task_description}")
            
            try:
                idx = int(input(f"\nEnter explanation number (1-{len(explanations)}): ")) - 1
                if 0 <= idx < len(explanations):
                    ExplanationDisplay.print_agent_explanation(explanations[idx])
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input.")
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")
