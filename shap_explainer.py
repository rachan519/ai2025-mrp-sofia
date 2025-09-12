"""
SHAP (SHapley Additive exPlanations) Implementation
Provides global and local explanations for learning plan predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import itertools
from scipy.special import comb
import warnings
warnings.filterwarnings('ignore')

from feature_extractor import LearningPlanFeatureExtractor, LearningPlanPredictor, FeatureVector
from models import CompleteLearningPlan


@dataclass
class SHAPExplanation:
    """Container for SHAP explanation results"""
    shap_values: Dict[str, float]
    base_value: float
    prediction: float
    feature_names: List[str]
    explanation_text: str
    global_importance: Optional[Dict[str, float]] = None


class SHAPExplainer:
    """SHAP explainer for learning plan predictions"""
    
    def __init__(self, 
                 feature_extractor: LearningPlanFeatureExtractor,
                 predictor: LearningPlanPredictor,
                 background_data: Optional[np.ndarray] = None,
                 max_features: int = 20):
        """
        Initialize SHAP explainer
        
        Args:
            feature_extractor: Feature extractor for learning plans
            predictor: Model to explain
            background_data: Background dataset for reference (optional)
            max_features: Maximum number of features to explain
        """
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        self.background_data = background_data
        self.max_features = max_features
        
    def explain_instance(self, 
                        learning_plan: CompleteLearningPlan,
                        prediction_function: Optional[Callable] = None) -> SHAPExplanation:
        """
        Explain a single learning plan instance using SHAP
        
        Args:
            learning_plan: Learning plan to explain
            prediction_function: Optional custom prediction function
            
        Returns:
            SHAPExplanation object with SHAP values and explanation
        """
        # Use the feature extractor's transform method to get consistent features
        instance_features = self.feature_extractor.transform(learning_plan)
        feature_names = self.feature_extractor.feature_names if hasattr(self.feature_extractor, 'feature_names') else [f'feature_{i}' for i in range(len(instance_features))]
        
        # Don't limit features for now to avoid dimension mismatch
        # if len(instance_features) > self.max_features:
        #     # Select most important features (simple heuristic)
        #     feature_importance = np.abs(instance_features)
        #     top_indices = np.argsort(feature_importance)[-self.max_features:]
        #     instance_features = instance_features[top_indices]
        #     feature_names = [feature_names[i] for i in top_indices]
        
        # Get prediction
        if prediction_function:
            prediction = prediction_function(instance_features.reshape(1, -1))[0]
        else:
            prediction = self.predictor.predict(instance_features.reshape(1, -1))[0]
        
        # Calculate SHAP values
        shap_values = self._calculate_shap_values(
            instance_features, feature_names, prediction_function
        )
        
        # Calculate base value (average prediction)
        base_value = self._calculate_base_value(instance_features, prediction_function)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            shap_values, base_value, prediction, learning_plan, feature_names
        )
        
        return SHAPExplanation(
            shap_values=shap_values,
            base_value=base_value,
            prediction=prediction,
            feature_names=feature_names,
            explanation_text=explanation_text
        )
    
    def _calculate_shap_values(self, 
                             instance_features: np.ndarray,
                             feature_names: List[str],
                             prediction_function: Optional[Callable] = None) -> Dict[str, float]:
        """Calculate SHAP values using exact computation for small feature sets"""
        num_features = len(instance_features)
        shap_values = {name: 0.0 for name in feature_names}
        
        # For small feature sets, use exact computation
        if num_features <= 10:
            shap_values = self._exact_shap_values(
                instance_features, feature_names, prediction_function
            )
        else:
            # For larger feature sets, use sampling approximation
            shap_values = self._sampling_shap_values(
                instance_features, feature_names, prediction_function
            )
        
        return shap_values
    
    def _exact_shap_values(self, 
                          instance_features: np.ndarray,
                          feature_names: List[str],
                          prediction_function: Optional[Callable] = None) -> Dict[str, float]:
        """Calculate exact SHAP values using the Shapley value formula"""
        num_features = len(instance_features)
        shap_values = {name: 0.0 for name in feature_names}
        
        # Calculate base prediction (all features absent)
        base_prediction = self._predict_with_mask(
            instance_features, np.zeros(num_features, dtype=bool), prediction_function
        )
        
        # Calculate SHAP values for each feature
        for i, feature_name in enumerate(feature_names):
            shap_value = 0.0
            
            # Sum over all possible subsets
            for subset_size in range(num_features):
                for subset in itertools.combinations(range(num_features), subset_size):
                    if i not in subset:
                        # Calculate marginal contribution
                        subset_mask = np.zeros(num_features, dtype=bool)
                        subset_mask[list(subset)] = True
                        
                        # With feature i
                        subset_with_i = subset_mask.copy()
                        subset_with_i[i] = True
                        pred_with = self._predict_with_mask(
                            instance_features, subset_with_i, prediction_function
                        )
                        
                        # Without feature i
                        pred_without = self._predict_with_mask(
                            instance_features, subset_mask, prediction_function
                        )
                        
                        # Marginal contribution
                        marginal_contribution = pred_with - pred_without
                        
                        # Weight by subset size
                        weight = 1.0 / (num_features * comb(num_features - 1, subset_size))
                        shap_value += weight * marginal_contribution
            
            shap_values[feature_name] = shap_value
        
        return shap_values
    
    def _sampling_shap_values(self, 
                            instance_features: np.ndarray,
                            feature_names: List[str],
                            prediction_function: Optional[Callable] = None,
                            num_samples: int = 1000) -> Dict[str, float]:
        """Calculate SHAP values using sampling approximation"""
        num_features = len(instance_features)
        shap_values = {name: 0.0 for name in feature_names}
        
        # Sample random feature subsets
        for _ in range(num_samples):
            # Random subset
            subset_mask = np.random.random(num_features) < 0.5
            
            for i, feature_name in enumerate(feature_names):
                # With feature i
                subset_with_i = subset_mask.copy()
                subset_with_i[i] = True
                pred_with = self._predict_with_mask(
                    instance_features, subset_with_i, prediction_function
                )
                
                # Without feature i
                pred_without = self._predict_with_mask(
                    instance_features, subset_mask, prediction_function
                )
                
                # Marginal contribution
                marginal_contribution = pred_with - pred_without
                shap_values[feature_name] += marginal_contribution / num_samples
        
        return shap_values
    
    def _predict_with_mask(self, 
                          instance_features: np.ndarray,
                          mask: np.ndarray,
                          prediction_function: Optional[Callable] = None) -> float:
        """Predict with a subset of features masked out"""
        # Create masked instance
        masked_instance = instance_features.copy()
        masked_instance[~mask.astype(bool)] = 0  # Set masked features to 0
        
        # Get prediction
        if prediction_function:
            return prediction_function(masked_instance.reshape(1, -1))[0]
        else:
            return self.predictor.predict(masked_instance.reshape(1, -1))[0]
    
    def _calculate_base_value(self, 
                            instance_features: np.ndarray,
                            prediction_function: Optional[Callable] = None) -> float:
        """Calculate base value (average prediction)"""
        if self.background_data is not None:
            # Use background data if available
            if prediction_function:
                predictions = prediction_function(self.background_data)
            else:
                predictions = self.predictor.predict(self.background_data)
            return np.mean(predictions)
        else:
            # Use zero vector as background
            zero_instance = np.zeros_like(instance_features)
            return self._predict_with_mask(zero_instance, np.zeros_like(instance_features), prediction_function)
    
    def _generate_explanation_text(self, 
                                 shap_values: Dict[str, float],
                                 base_value: float,
                                 prediction: float,
                                 learning_plan: CompleteLearningPlan,
                                 feature_names: List[str]) -> str:
        """Generate human-readable explanation text"""
        explanation_parts = []
        
        # Overall prediction breakdown
        explanation_parts.append(f"Learning Plan Effectiveness Prediction: {prediction:.3f}")
        explanation_parts.append(f"Base Effectiveness: {base_value:.3f}")
        explanation_parts.append(f"Feature Contributions: {prediction - base_value:.3f}")
        
        # Sort features by absolute SHAP value
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Top positive contributions
        positive_features = [(name, value) for name, value in sorted_features if value > 0.01]
        if positive_features:
            explanation_parts.append("\nFeatures that increase effectiveness:")
            for name, value in positive_features[:5]:  # Top 5
                explanation_parts.append(f"  • {self._format_feature_name(name)}: +{value:.3f}")
        
        # Top negative contributions
        negative_features = [(name, value) for name, value in sorted_features if value < -0.01]
        if negative_features:
            explanation_parts.append("\nFeatures that decrease effectiveness:")
            for name, value in negative_features[:5]:  # Top 5
                explanation_parts.append(f"  • {self._format_feature_name(name)}: {value:.3f}")
        
        # Learning plan specific insights
        insights = self._generate_learning_plan_insights(shap_values, learning_plan)
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
                                       shap_values: Dict[str, float],
                                       learning_plan: CompleteLearningPlan) -> List[str]:
        """Generate specific insights based on SHAP values"""
        insights = []
        
        # Topic complexity insights
        if 'topic_length' in shap_values:
            if shap_values['topic_length'] > 0.1:
                insights.append(f"  • The specific topic '{learning_plan.user_input.topic}' is well-defined")
            elif shap_values['topic_length'] < -0.1:
                insights.append(f"  • The topic '{learning_plan.user_input.topic}' could be more specific")
        
        # Background insights
        if 'background_has_experience' in shap_values:
            if shap_values['background_has_experience'] > 0.1:
                insights.append("  • Your relevant experience significantly improves the learning plan")
            elif shap_values['background_has_experience'] < -0.1:
                insights.append("  • Consider adding more relevant experience to your background")
        
        # Learning format insights
        format_features = [f for f in shap_values.keys() if 'preferred_format' in f]
        if format_features:
            best_format = max(format_features, key=lambda x: shap_values[x])
            if shap_values[best_format] > 0.1:
                format_name = best_format.replace('preferred_format_', '').title()
                insights.append(f"  • {format_name} format is optimal for your learning style")
        
        # Duration insights
        if 'duration_weeks' in shap_values:
            if shap_values['duration_weeks'] > 0.1:
                insights.append(f"  • The timeline of {learning_plan.topic_plan.estimated_duration} is well-calibrated")
            elif shap_values['duration_weeks'] < -0.1:
                insights.append("  • Consider adjusting the learning timeline for better results")
        
        # Gap analysis insights
        if 'num_gaps' in shap_values:
            if shap_values['num_gaps'] > 0.1:
                insights.append(f"  • Identifying {len(learning_plan.knowledge_gap.identified_gaps)} specific gaps provides excellent structure")
            elif shap_values['num_gaps'] < -0.1:
                insights.append("  • More detailed gap analysis would improve the learning plan")
        
        return insights
    
    def explain_multiple_instances(self, 
                                 learning_plans: List[CompleteLearningPlan],
                                 prediction_function: Optional[Callable] = None) -> List[SHAPExplanation]:
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
        """Get global feature importance by averaging SHAP values"""
        explanations = self.explain_multiple_instances(learning_plans, prediction_function)
        
        if not explanations:
            return {}
        
        # Average SHAP values across all explanations
        all_features = set()
        for exp in explanations:
            all_features.update(exp.shap_values.keys())
        
        global_importance = {}
        for feature in all_features:
            shap_values = [exp.shap_values.get(feature, 0) for exp in explanations]
            global_importance[feature] = np.mean(np.abs(shap_values))  # Use absolute values for importance
        
        # Sort by importance
        global_importance = dict(sorted(
            global_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return global_importance
    
    def create_summary_plot_data(self, 
                               explanations: List[SHAPExplanation]) -> Dict[str, Any]:
        """Create data for summary plots"""
        if not explanations:
            return {}
        
        # Collect all SHAP values
        all_features = set()
        for exp in explanations:
            all_features.update(exp.shap_values.keys())
        
        # Create matrix of SHAP values
        shap_matrix = []
        feature_names = sorted(all_features)
        
        for exp in explanations:
            row = [exp.shap_values.get(feature, 0) for feature in feature_names]
            shap_matrix.append(row)
        
        return {
            'shap_values': np.array(shap_matrix),
            'feature_names': feature_names,
            'predictions': [exp.prediction for exp in explanations],
            'base_values': [exp.base_value for exp in explanations]
        }


class SHAPVisualizer:
    """Visualization tools for SHAP explanations"""
    
    @staticmethod
    def plot_waterfall(explanation: SHAPExplanation, 
                      top_n: int = 10,
                      figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot waterfall chart for SHAP explanation"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return
        
        # Get top N features
        sorted_features = sorted(explanation.shap_values.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        if not sorted_features:
            print("No features to plot")
            return
        
        features, values = zip(*sorted_features)
        
        # Create waterfall chart
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate cumulative values
        cumulative = explanation.base_value
        y_pos = np.arange(len(features))
        
        # Plot bars
        colors = ['green' if v > 0 else 'red' for v in values]
        bars = ax.barh(y_pos, values, color=colors, alpha=0.7)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left' if width > 0 else 'right', va='center')
        
        # Add base value line
        ax.axvline(explanation.base_value, color='black', linestyle='--', alpha=0.7, label='Base Value')
        
        # Add prediction line
        ax.axvline(explanation.prediction, color='blue', linestyle='-', alpha=0.7, label='Prediction')
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
        ax.set_xlabel('SHAP Value')
        ax.set_title('SHAP Waterfall Plot')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_summary(explanations: List[SHAPExplanation],
                    top_n: int = 15,
                    figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot summary plot for multiple explanations"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return
        
        if not explanations:
            print("No explanations to plot")
            return
        
        # Get global importance
        all_features = set()
        for exp in explanations:
            all_features.update(exp.shap_values.keys())
        
        global_importance = {}
        for feature in all_features:
            values = [exp.shap_values.get(feature, 0) for exp in explanations]
            global_importance[feature] = np.mean(np.abs(values))
        
        # Get top N features
        top_features = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if not top_features:
            print("No features to plot")
            return
        
        features, importances = zip(*top_features)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(features)), importances, alpha=0.7)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('SHAP Summary Plot')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def print_explanation(explanation: SHAPExplanation) -> None:
        """Print formatted SHAP explanation"""
        print("=" * 80)
        print("SHAP EXPLANATION")
        print("=" * 80)
        print(explanation.explanation_text)
        print("\n" + "=" * 80)
        print("DETAILED SHAP VALUES")
        print("=" * 80)
        
        # Sort by absolute SHAP value
        sorted_features = sorted(explanation.shap_values.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
        
        for feature, value in sorted_features:
            direction = "increases" if value > 0 else "decreases"
            print(f"{feature:30} | {value:8.4f} | {direction} effectiveness")
        
        print(f"\nBase Value: {explanation.base_value:.4f}")
        print(f"Prediction: {explanation.prediction:.4f}")
        print(f"Sum of SHAP values: {sum(explanation.shap_values.values()):.4f}")


def demonstrate_shap():
    """Demonstrate SHAP explainer functionality"""
    print("SHAP (SHapley Additive exPlanations) Demo")
    print("=" * 70)
    
    # Create sample data
    feature_extractor = LearningPlanFeatureExtractor()
    learning_plans = feature_extractor.create_sample_learning_plans()
    predictor = create_sample_predictor()
    
    # Create SHAP explainer
    shap_explainer = SHAPExplainer(feature_extractor, predictor)
    
    # Explain first learning plan
    print("Explaining Learning Plan 1:")
    print("-" * 40)
    explanation1 = shap_explainer.explain_instance(learning_plans[0])
    SHAPVisualizer.print_explanation(explanation1)
    
    print("\n" + "=" * 70)
    print("Explaining Learning Plan 2:")
    print("-" * 40)
    explanation2 = shap_explainer.explain_instance(learning_plans[1])
    SHAPVisualizer.print_explanation(explanation2)
    
    # Global feature importance
    print("\n" + "=" * 70)
    print("GLOBAL FEATURE IMPORTANCE")
    print("=" * 70)
    global_importance = shap_explainer.get_global_feature_importance(learning_plans)
    
    for feature, importance in list(global_importance.items())[:10]:
        print(f"{feature:30} | {importance:8.4f}")


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


if __name__ == "__main__":
    demonstrate_shap()
