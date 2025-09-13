"""
LIME (Local Interpretable Model-agnostic Explanations) Implementation
Provides local explanations for individual learning plan predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
import random
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

from feature_extractor import LearningPlanFeatureExtractor, LearningPlanPredictor, FeatureVector
from models import CompleteLearningPlan


@dataclass
class LIMEExplanation:
    """Container for LIME explanation results"""
    feature_importance: Dict[str, float]
    prediction: float
    confidence: float
    local_model_score: float
    feature_names: List[str]
    explanation_text: str


class LIMEExplainer:
    """LIME explainer for learning plan predictions"""
    
    def __init__(self, 
                 feature_extractor: LearningPlanFeatureExtractor,
                 predictor: LearningPlanPredictor,
                 kernel_width: float = 0.75,
                 num_samples: int = 5000,
                 random_state: int = 42):
        """
        Initialize LIME explainer
        
        Args:
            feature_extractor: Feature extractor for learning plans
            predictor: Model to explain
            kernel_width: Width of the kernel for distance weighting
            num_samples: Number of samples to generate for local explanation
            random_state: Random state for reproducibility
        """
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        self.kernel_width = kernel_width
        self.num_samples = num_samples
        self.random_state = random_state
        self.random_state_obj = np.random.RandomState(random_state)
        
    def explain_instance(self, 
                        learning_plan: CompleteLearningPlan,
                        prediction_function: Optional[Callable] = None) -> LIMEExplanation:
        """
        Explain a single learning plan instance using LIME
        
        Args:
            learning_plan: Learning plan to explain
            prediction_function: Optional custom prediction function
            
        Returns:
            LIMEExplanation object with feature importance and explanation
        """
        # Use the feature extractor's transform method to get consistent features
        instance_features = self.feature_extractor.transform(learning_plan)
        feature_names = self.feature_extractor.feature_names if hasattr(self.feature_extractor, 'feature_names') else [f'feature_{i}' for i in range(len(instance_features))]
        
        # Get prediction
        if prediction_function:
            prediction = prediction_function(instance_features.reshape(1, -1))[0]
        else:
            prediction = self.predictor.predict(instance_features.reshape(1, -1))[0]
        
        # Generate perturbed samples
        perturbed_samples, weights = self._generate_perturbed_samples(instance_features)
        
        # Get predictions for perturbed samples
        if prediction_function:
            perturbed_predictions = prediction_function(perturbed_samples)
        else:
            perturbed_predictions = self.predictor.predict(perturbed_samples)
        
        # Fit local linear model
        local_model, local_score = self._fit_local_model(
            perturbed_samples, perturbed_predictions, weights
        )
        
        # Extract feature importance
        feature_importance = self._extract_feature_importance(
            local_model, feature_names
        )
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            feature_importance, prediction, learning_plan
        )
        
        return LIMEExplanation(
            feature_importance=feature_importance,
            prediction=prediction,
            confidence=local_score,
            local_model_score=local_score,
            feature_names=feature_names,
            explanation_text=explanation_text
        )
    
    def _generate_perturbed_samples(self, instance_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate perturbed samples around the instance"""
        num_features = len(instance_features)
        perturbed_samples = np.zeros((self.num_samples, num_features))
        
        # Generate random perturbations
        for i in range(self.num_samples):
            # Randomly select features to perturb
            mask = self.random_state_obj.random(num_features) < 0.5
            perturbed_samples[i] = instance_features.copy()
            
            # Perturb selected features
            for j in range(num_features):
                if mask[j]:
                    # Add Gaussian noise
                    noise = self.random_state_obj.normal(0, 0.1)
                    perturbed_samples[i][j] += noise
                    
                    # Ensure categorical features stay in valid range
                    if perturbed_samples[i][j] < 0:
                        perturbed_samples[i][j] = 0
                    elif perturbed_samples[i][j] > 1:
                        perturbed_samples[i][j] = 1
        
        # Calculate distances and weights
        distances = pairwise_distances(perturbed_samples, instance_features.reshape(1, -1)).flatten()
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        
        return perturbed_samples, weights
    
    def _fit_local_model(self, 
                        samples: np.ndarray, 
                        predictions: np.ndarray, 
                        weights: np.ndarray) -> Tuple[Ridge, float]:
        """Fit local linear model to explain the prediction"""
        # Use Ridge regression for stability
        local_model = Ridge(alpha=1.0)
        local_model.fit(samples, predictions, sample_weight=weights)
        
        # Calculate R² score
        local_predictions = local_model.predict(samples)
        ss_res = np.sum(weights * (predictions - local_predictions) ** 2)
        ss_tot = np.sum(weights * (predictions - np.average(predictions, weights=weights)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return local_model, r2_score
    
    def _extract_feature_importance(self, 
                                  local_model: Ridge, 
                                  feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from local model coefficients"""
        coefficients = local_model.coef_
        feature_importance = {}
        
        for i, name in enumerate(feature_names):
            feature_importance[name] = float(coefficients[i])
        
        # Sort by absolute importance
        feature_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        ))
        
        return feature_importance
    
    def _generate_explanation_text(self, 
                                 feature_importance: Dict[str, float],
                                 prediction: float,
                                 learning_plan: CompleteLearningPlan) -> str:
        """Generate human-readable explanation text"""
        explanation_parts = []
        
        # Overall prediction
        explanation_parts.append(f"Predicted learning plan effectiveness: {prediction:.2f}")
        
        # Top positive features
        positive_features = [(name, importance) for name, importance in feature_importance.items() 
                           if importance > 0.1]
        if positive_features:
            explanation_parts.append("\nFactors that increase effectiveness:")
            for name, importance in positive_features[:5]:  # Top 5
                explanation_parts.append(f"  • {self._format_feature_name(name)}: +{importance:.3f}")
        
        # Top negative features
        negative_features = [(name, importance) for name, importance in feature_importance.items() 
                           if importance < -0.1]
        if negative_features:
            explanation_parts.append("\nFactors that decrease effectiveness:")
            for name, importance in negative_features[:5]:  # Top 5
                explanation_parts.append(f"  • {self._format_feature_name(name)}: {importance:.3f}")
        
        # Learning plan specific insights
        insights = self._generate_learning_plan_insights(feature_importance, learning_plan)
        if insights:
            explanation_parts.append(f"\nSpecific insights for this learning plan:")
            explanation_parts.extend(insights)
        
        return "\n".join(explanation_parts)
    
    def _format_feature_name(self, feature_name: str) -> str:
        """Format feature name for human readability"""
        # Replace underscores with spaces and capitalize
        formatted = feature_name.replace('_', ' ').title()
        
        # Special cases
        replacements = {
            'Num ': 'Number of ',
            'Avg ': 'Average ',
            'Has ': 'Contains ',
            'Preferred Format': 'Learning Format',
            'Current Level': 'Current Knowledge Level',
            'Target Level': 'Target Knowledge Level',
            'Background Length': 'Background Description Length',
            'Topic Length': 'Topic Description Length',
            'Duration Weeks': 'Duration (Weeks)',
            'Duration Hours Per Week': 'Study Time (Hours/Week)'
        }
        
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)
        
        return formatted
    
    def _generate_learning_plan_insights(self, 
                                       feature_importance: Dict[str, float],
                                       learning_plan: CompleteLearningPlan) -> List[str]:
        """Generate specific insights based on the learning plan"""
        insights = []
        
        # Topic complexity insights
        if 'topic_length' in feature_importance:
            if feature_importance['topic_length'] > 0.1:
                insights.append(f"  • The topic '{learning_plan.user_input.topic}' is well-defined and specific")
            elif feature_importance['topic_length'] < -0.1:
                insights.append(f"  • The topic '{learning_plan.user_input.topic}' might be too broad or vague")
        
        # Background insights
        if 'background_has_experience' in feature_importance:
            if feature_importance['background_has_experience'] > 0.1:
                insights.append("  • Your background shows relevant experience, which is beneficial")
            elif feature_importance['background_has_experience'] < -0.1:
                insights.append("  • Consider highlighting more relevant experience in your background")
        
        # Learning format insights
        format_features = [f for f in feature_importance.keys() if 'preferred_format' in f]
        if format_features:
            best_format = max(format_features, key=lambda x: feature_importance[x])
            if feature_importance[best_format] > 0.1:
                format_name = best_format.replace('preferred_format_', '').title()
                insights.append(f"  • {format_name} format is well-suited for this learning plan")
        
        # Duration insights
        if 'duration_weeks' in feature_importance:
            if feature_importance['duration_weeks'] > 0.1:
                insights.append(f"  • The estimated duration of {learning_plan.topic_plan.estimated_duration} is appropriate")
            elif feature_importance['duration_weeks'] < -0.1:
                insights.append("  • Consider adjusting the learning timeline for better effectiveness")
        
        # Gap analysis insights
        if 'num_gaps' in feature_importance:
            if feature_importance['num_gaps'] > 0.1:
                insights.append(f"  • Identifying {len(learning_plan.knowledge_gap.identified_gaps)} knowledge gaps provides good structure")
            elif feature_importance['num_gaps'] < -0.1:
                insights.append("  • Consider identifying more specific knowledge gaps for better planning")
        
        return insights
    
    def explain_multiple_instances(self, 
                                 learning_plans: List[CompleteLearningPlan],
                                 prediction_function: Optional[Callable] = None) -> List[LIMEExplanation]:
        """Explain multiple learning plan instances"""
        explanations = []
        
        for plan in learning_plans:
            try:
                explanation = self.explain_instance(plan, prediction_function)
                explanations.append(explanation)
            except Exception as e:
                print(f"Warning: Could not explain learning plan: {e}")
                continue
        
        return explanations
    
    def get_global_feature_importance(self, 
                                    learning_plans: List[CompleteLearningPlan],
                                    prediction_function: Optional[Callable] = None) -> Dict[str, float]:
        """Get global feature importance by averaging local explanations"""
        explanations = self.explain_multiple_instances(learning_plans, prediction_function)
        
        if not explanations:
            return {}
        
        # Average feature importance across all explanations
        all_features = set()
        for exp in explanations:
            all_features.update(exp.feature_importance.keys())
        
        global_importance = {}
        for feature in all_features:
            importances = [exp.feature_importance.get(feature, 0) for exp in explanations]
            global_importance[feature] = np.mean(importances)
        
        # Sort by absolute importance
        global_importance = dict(sorted(
            global_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        return global_importance


class LIMEVisualizer:
    """Visualization tools for LIME explanations"""
    
    @staticmethod
    def plot_feature_importance(explanation: LIMEExplanation, 
                              top_n: int = 10,
                              figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot feature importance for a LIME explanation"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return
        
        # Get top N features
        top_features = list(explanation.feature_importance.items())[:top_n]
        features, importances = zip(*top_features)
        
        # Create plot
        plt.figure(figsize=figsize)
        colors = ['green' if imp > 0 else 'red' for imp in importances]
        
        plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
        plt.yticks(range(len(features)), [f.replace('_', ' ').title() for f in features])
        plt.xlabel('Feature Importance')
        plt.title('LIME Feature Importance')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(importances):
            plt.text(v + (0.01 if v > 0 else -0.01), i, f'{v:.3f}', 
                    va='center', ha='left' if v > 0 else 'right')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def print_explanation(explanation: LIMEExplanation) -> None:
        """Print formatted LIME explanation"""
        print("=" * 80)
        print("LIME EXPLANATION")
        print("=" * 80)
        print(explanation.explanation_text)
        print("\n" + "=" * 80)
        print("DETAILED FEATURE IMPORTANCE")
        print("=" * 80)
        
        for feature, importance in explanation.feature_importance.items():
            direction = "increases" if importance > 0 else "decreases"
            print(f"{feature:30} | {importance:8.4f} | {direction} effectiveness")
        
        print(f"\nLocal Model R² Score: {explanation.local_model_score:.4f}")
        print(f"Prediction: {explanation.prediction:.4f}")


def create_sample_predictor() -> LearningPlanPredictor:
    """Create a sample predictor for testing"""
    predictor = LearningPlanPredictor()
    
    # Create sample data
    feature_extractor = LearningPlanFeatureExtractor()
    learning_plans = feature_extractor.create_sample_learning_plans()
    
    # Extract features
    X, feature_names = feature_extractor.fit_transform(learning_plans)
    
    # Create synthetic target values (learning plan effectiveness scores)
    y = np.array([0.8, 0.9])  # Sample effectiveness scores
    
    # Fit predictor
    predictor.fit(X, y)
    
    return predictor


def demonstrate_lime():
    """Demonstrate LIME explainer functionality"""
    print("LIME (Local Interpretable Model-agnostic Explanations) Demo")
    print("=" * 70)
    
    # Create sample data
    feature_extractor = LearningPlanFeatureExtractor()
    learning_plans = feature_extractor.create_sample_learning_plans()
    predictor = create_sample_predictor()
    
    # Create LIME explainer
    lime_explainer = LIMEExplainer(feature_extractor, predictor)
    
    # Explain first learning plan
    print("Explaining Learning Plan 1:")
    print("-" * 40)
    explanation1 = lime_explainer.explain_instance(learning_plans[0])
    LIMEVisualizer.print_explanation(explanation1)
    
    print("\n" + "=" * 70)
    print("Explaining Learning Plan 2:")
    print("-" * 40)
    explanation2 = lime_explainer.explain_instance(learning_plans[1])
    LIMEVisualizer.print_explanation(explanation2)
    
    # Global feature importance
    print("\n" + "=" * 70)
    print("GLOBAL FEATURE IMPORTANCE")
    print("=" * 70)
    global_importance = lime_explainer.get_global_feature_importance(learning_plans)
    
    for feature, importance in list(global_importance.items())[:10]:
        print(f"{feature:30} | {importance:8.4f}")


if __name__ == "__main__":
    demonstrate_lime()
