"""
Integrated Explainability System
Combines Chain-of-Thought/ReAct patterns with LIME/SHAP for comprehensive explanations.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from models import CompleteLearningPlan, AgentExplanation
from feature_extractor import LearningPlanFeatureExtractor, LearningPlanPredictor
from lime_explainer import LIMEExplainer, LIMEExplanation, LIMEVisualizer
from shap_explainer import SHAPExplainer, SHAPExplanation, SHAPVisualizer
from explanation_logger import ExplanationLogger


@dataclass
class IntegratedExplanation:
    """Container for integrated explanation results"""
    agent_explanations: List[AgentExplanation]
    lime_explanation: Optional[LIMEExplanation]
    shap_explanation: Optional[SHAPExplanation]
    global_feature_importance: Dict[str, float]
    learning_plan: CompleteLearningPlan
    prediction: float
    confidence: float
    explanation_summary: str
    timestamp: datetime


class IntegratedExplainabilitySystem:
    """Integrated explainability system combining multiple explanation methods"""
    
    def __init__(self, 
                 feature_extractor: LearningPlanFeatureExtractor,
                 predictor: LearningPlanPredictor,
                 explanation_logger: ExplanationLogger):
        """
        Initialize integrated explainability system
        
        Args:
            feature_extractor: Feature extractor for learning plans
            predictor: Model to explain
            explanation_logger: Logger for agent explanations
        """
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        self.explanation_logger = explanation_logger
        
        # Initialize explainers
        self.lime_explainer = LIMEExplainer(feature_extractor, predictor)
        self.shap_explainer = SHAPExplainer(feature_extractor, predictor)
        
    def explain_learning_plan(self, 
                            learning_plan: CompleteLearningPlan,
                            agent_explanations: List[AgentExplanation],
                            include_lime: bool = True,
                            include_shap: bool = True) -> IntegratedExplanation:
        """
        Generate comprehensive explanation for a learning plan
        
        Args:
            learning_plan: Learning plan to explain
            agent_explanations: Agent explanations from the workflow
            include_lime: Whether to include LIME explanations
            include_shap: Whether to include SHAP explanations
            
        Returns:
            IntegratedExplanation with all explanation methods
        """
        # Get prediction
        features = self.feature_extractor.transform(learning_plan)
        prediction = self.predictor.predict(features.reshape(1, -1))[0]
        
        # Generate LIME explanation
        lime_explanation = None
        if include_lime:
            try:
                lime_explanation = self.lime_explainer.explain_instance(learning_plan)
            except Exception as e:
                print(f"Warning: Could not generate LIME explanation: {e}")
        
        # Generate SHAP explanation
        shap_explanation = None
        if include_shap:
            try:
                shap_explanation = self.shap_explainer.explain_instance(learning_plan)
            except Exception as e:
                print(f"Warning: Could not generate SHAP explanation: {e}")
        
        # Get global feature importance
        global_importance = self._get_global_feature_importance(
            learning_plan, lime_explanation, shap_explanation
        )
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(
            agent_explanations, lime_explanation, shap_explanation
        )
        
        # Generate integrated explanation summary
        explanation_summary = self._generate_integrated_summary(
            learning_plan, agent_explanations, lime_explanation, shap_explanation, 
            global_importance, prediction, confidence
        )
        
        return IntegratedExplanation(
            agent_explanations=agent_explanations,
            lime_explanation=lime_explanation,
            shap_explanation=shap_explanation,
            global_feature_importance=global_importance,
            learning_plan=learning_plan,
            prediction=prediction,
            confidence=confidence,
            explanation_summary=explanation_summary,
            timestamp=datetime.now()
        )
    
    def _get_global_feature_importance(self, 
                                     learning_plan: CompleteLearningPlan,
                                     lime_explanation: Optional[LIMEExplanation],
                                     shap_explanation: Optional[SHAPExplanation]) -> Dict[str, float]:
        """Get global feature importance from available explanations"""
        global_importance = {}
        
        # Combine LIME and SHAP importance if available
        if lime_explanation and shap_explanation:
            # Average the importance scores
            all_features = set(lime_explanation.feature_importance.keys())
            all_features.update(shap_explanation.shap_values.keys())
            
            for feature in all_features:
                lime_imp = abs(lime_explanation.feature_importance.get(feature, 0))
                shap_imp = abs(shap_explanation.shap_values.get(feature, 0))
                global_importance[feature] = (lime_imp + shap_imp) / 2
                
        elif lime_explanation:
            global_importance = {k: abs(v) for k, v in lime_explanation.feature_importance.items()}
        elif shap_explanation:
            global_importance = {k: abs(v) for k, v in shap_explanation.shap_values.items()}
        
        # Sort by importance
        global_importance = dict(sorted(
            global_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return global_importance
    
    def _calculate_confidence(self, 
                            agent_explanations: List[AgentExplanation],
                            lime_explanation: Optional[LIMEExplanation],
                            shap_explanation: Optional[SHAPExplanation]) -> float:
        """Calculate overall confidence from all explanation methods"""
        confidences = []
        
        # Agent confidence
        if agent_explanations:
            agent_confidence = np.mean([exp.confidence_score for exp in agent_explanations])
            confidences.append(agent_confidence)
        
        # LIME confidence
        if lime_explanation:
            confidences.append(lime_explanation.confidence)
        
        # SHAP confidence (based on prediction stability)
        if shap_explanation:
            # Simple heuristic: higher confidence if SHAP values are more stable
            shap_values = list(shap_explanation.shap_values.values())
            if shap_values:
                shap_confidence = 1.0 - (np.std(shap_values) / (np.mean(np.abs(shap_values)) + 1e-8))
                confidences.append(max(0, min(1, shap_confidence)))
        
        return np.mean(confidences) if confidences else 0.5
    
    def _generate_integrated_summary(self, 
                                   learning_plan: CompleteLearningPlan,
                                   agent_explanations: List[AgentExplanation],
                                   lime_explanation: Optional[LIMEExplanation],
                                   shap_explanation: Optional[SHAPExplanation],
                                   global_importance: Dict[str, float],
                                   prediction: float,
                                   confidence: float) -> str:
        """Generate comprehensive explanation summary"""
        summary_parts = []
        
        # Header
        summary_parts.append("=" * 80)
        summary_parts.append("COMPREHENSIVE LEARNING PLAN EXPLANATION")
        summary_parts.append("=" * 80)
        
        # Learning plan overview
        summary_parts.append(f"Topic: {learning_plan.user_input.topic}")
        summary_parts.append(f"Background: {learning_plan.user_input.background[:100]}...")
        summary_parts.append(f"Preferred Format: {learning_plan.user_input.preferred_format.value}")
        summary_parts.append(f"Predicted Effectiveness: {prediction:.3f}")
        summary_parts.append(f"Overall Confidence: {confidence:.3f}")
        
        # Agent explanations summary
        if agent_explanations:
            summary_parts.append(f"\nAGENT DECISION PROCESS ({len(agent_explanations)} agents):")
            summary_parts.append("-" * 50)
            
            for i, exp in enumerate(agent_explanations, 1):
                summary_parts.append(f"{i}. {exp.agent_name}: {exp.task_description}")
                summary_parts.append(f"   Confidence: {exp.confidence_score:.3f}")
                if exp.chain_of_thought:
                    summary_parts.append(f"   Key Decision: {exp.chain_of_thought.decision}")
        
        # Feature importance summary
        if global_importance:
            summary_parts.append(f"\nKEY FACTORS AFFECTING EFFECTIVENESS:")
            summary_parts.append("-" * 50)
            
            for i, (feature, importance) in enumerate(list(global_importance.items())[:10], 1):
                formatted_name = feature.replace('_', ' ').title()
                summary_parts.append(f"{i:2d}. {formatted_name:30} | {importance:.3f}")
        
        # LIME insights
        if lime_explanation:
            summary_parts.append(f"\nLIME LOCAL EXPLANATION:")
            summary_parts.append("-" * 50)
            summary_parts.append(f"Local Model Confidence: {lime_explanation.confidence:.3f}")
            
            # Top positive and negative features
            positive_features = [(k, v) for k, v in lime_explanation.feature_importance.items() if v > 0.1]
            negative_features = [(k, v) for k, v in lime_explanation.feature_importance.items() if v < -0.1]
            
            if positive_features:
                summary_parts.append("Top Positive Factors:")
                for feature, value in positive_features[:3]:
                    summary_parts.append(f"  • {feature.replace('_', ' ').title()}: +{value:.3f}")
            
            if negative_features:
                summary_parts.append("Top Negative Factors:")
                for feature, value in negative_features[:3]:
                    summary_parts.append(f"  • {feature.replace('_', ' ').title()}: {value:.3f}")
        
        # SHAP insights
        if shap_explanation:
            summary_parts.append(f"\nSHAP GLOBAL EXPLANATION:")
            summary_parts.append("-" * 50)
            summary_parts.append(f"Base Value: {shap_explanation.base_value:.3f}")
            summary_parts.append(f"Feature Contributions: {shap_explanation.prediction - shap_explanation.base_value:.3f}")
            
            # Top contributions
            sorted_shap = sorted(shap_explanation.shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
            summary_parts.append("Top Feature Contributions:")
            for feature, value in sorted_shap[:5]:
                direction = "increases" if value > 0 else "decreases"
                summary_parts.append(f"  • {feature.replace('_', ' ').title()}: {value:.3f} ({direction})")
        
        # Recommendations
        recommendations = self._generate_recommendations(
            learning_plan, global_importance, prediction, confidence
        )
        if recommendations:
            summary_parts.append(f"\nRECOMMENDATIONS:")
            summary_parts.append("-" * 50)
            for i, rec in enumerate(recommendations, 1):
                summary_parts.append(f"{i}. {rec}")
        
        summary_parts.append("\n" + "=" * 80)
        
        return "\n".join(summary_parts)
    
    def _generate_recommendations(self, 
                                learning_plan: CompleteLearningPlan,
                                global_importance: Dict[str, float],
                                prediction: float,
                                confidence: float) -> List[str]:
        """Generate actionable recommendations based on explanations"""
        recommendations = []
        
        # Prediction-based recommendations
        if prediction < 0.6:
            recommendations.append("Consider refining the learning plan structure for better effectiveness")
        elif prediction > 0.8:
            recommendations.append("This learning plan shows high potential for success")
        
        # Confidence-based recommendations
        if confidence < 0.6:
            recommendations.append("Gather more information to improve plan confidence")
        
        # Feature-based recommendations
        if 'topic_length' in global_importance and global_importance['topic_length'] > 0.1:
            if len(learning_plan.user_input.topic.split()) < 3:
                recommendations.append("Consider making the learning topic more specific and detailed")
        
        if 'background_has_experience' in global_importance and global_importance['background_has_experience'] > 0.1:
            if not any(word in learning_plan.user_input.background.lower() 
                      for word in ['experience', 'worked', 'used', 'familiar']):
                recommendations.append("Highlight relevant experience in your background for better planning")
        
        if 'duration_weeks' in global_importance and global_importance['duration_weeks'] > 0.1:
            recommendations.append("The estimated timeline appears well-calibrated for the topic complexity")
        
        if 'num_gaps' in global_importance and global_importance['num_gaps'] > 0.1:
            if len(learning_plan.knowledge_gap.identified_gaps) < 3:
                recommendations.append("Consider identifying more specific knowledge gaps for better structure")
        
        # Learning format recommendations
        format_features = [f for f in global_importance.keys() if 'preferred_format' in f]
        if format_features:
            best_format = max(format_features, key=lambda x: global_importance[x])
            if global_importance[best_format] > 0.1:
                format_name = best_format.replace('preferred_format_', '').title()
                recommendations.append(f"Your choice of {format_name} format is well-suited for this topic")
        
        return recommendations
    
    def save_explanation(self, explanation: IntegratedExplanation, filename: str) -> str:
        """Save integrated explanation to file"""
        import json
        
        # Convert to serializable format
        explanation_data = {
            'learning_plan': explanation.learning_plan.model_dump(),
            'prediction': explanation.prediction,
            'confidence': explanation.confidence,
            'explanation_summary': explanation.explanation_summary,
            'timestamp': explanation.timestamp.isoformat(),
            'global_feature_importance': explanation.global_feature_importance,
            'agent_explanations_count': len(explanation.agent_explanations),
            'has_lime': explanation.lime_explanation is not None,
            'has_shap': explanation.shap_explanation is not None
        }
        
        # Add LIME data if available
        if explanation.lime_explanation:
            explanation_data['lime'] = {
                'feature_importance': explanation.lime_explanation.feature_importance,
                'prediction': explanation.lime_explanation.prediction,
                'confidence': explanation.lime_explanation.confidence
            }
        
        # Add SHAP data if available
        if explanation.shap_explanation:
            explanation_data['shap'] = {
                'shap_values': explanation.shap_explanation.shap_values,
                'base_value': explanation.shap_explanation.base_value,
                'prediction': explanation.shap_explanation.prediction
            }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(explanation_data, f, indent=2, default=str)
        
        return filename
    
    def print_explanation(self, explanation: IntegratedExplanation) -> None:
        """Print formatted integrated explanation"""
        print(explanation.explanation_summary)
    
    def visualize_explanations(self, explanation: IntegratedExplanation) -> None:
        """Visualize explanations using available tools"""
        if explanation.lime_explanation:
            print("\nLIME Visualization:")
            LIMEVisualizer.print_explanation(explanation.lime_explanation)
        
        if explanation.shap_explanation:
            print("\nSHAP Visualization:")
            SHAPVisualizer.print_explanation(explanation.shap_explanation)


def create_integrated_system() -> IntegratedExplainabilitySystem:
    """Create an integrated explainability system for testing"""
    # Create components
    feature_extractor = LearningPlanFeatureExtractor()
    predictor = LearningPlanPredictor()
    explanation_logger = ExplanationLogger()
    
    # Create sample data and fit predictor
    learning_plans = feature_extractor.create_sample_learning_plans()
    X, feature_names = feature_extractor.fit_transform(learning_plans)
    y = np.array([0.8, 0.9])  # Sample effectiveness scores
    predictor.fit(X, y)
    
    return IntegratedExplainabilitySystem(feature_extractor, predictor, explanation_logger)


def demonstrate_integrated_explainability():
    """Demonstrate integrated explainability system"""
    print("INTEGRATED EXPLAINABILITY SYSTEM DEMO")
    print("=" * 70)
    
    # Create system
    system = create_integrated_system()
    
    # Create sample learning plan
    learning_plans = system.feature_extractor.create_sample_learning_plans()
    learning_plan = learning_plans[0]
    
    # Create sample agent explanations
    from models import AgentExplanation
    agent_explanations = [
        AgentExplanation(
            agent_name="GapAnalysisAgent",
            task_description="Identify knowledge gaps",
            input_summary={"topic": "Python Programming"},
            output_summary={"gaps": ["Basic concepts", "Syntax"]},
            decision_factors=["Topic complexity", "User background"],
            alternative_approaches=["Generic analysis"],
            limitations=["Self-reported background"],
            confidence_score=0.8,
            processing_time=2.3
        )
    ]
    
    # Generate integrated explanation
    explanation = system.explain_learning_plan(
        learning_plan, 
        agent_explanations,
        include_lime=True,
        include_shap=True
    )
    
    # Print explanation
    system.print_explanation(explanation)
    
    # Save explanation
    filename = system.save_explanation(explanation, "integrated_explanation.json")
    print(f"\nExplanation saved to: {filename}")
    
    return explanation


if __name__ == "__main__":
    demonstrate_integrated_explainability()
