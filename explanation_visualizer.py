"""
Advanced Visualization Tools for LIME/SHAP Explanations
Provides comprehensive visualization capabilities for model-agnostic explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from lime_explainer import LIMEExplanation
from shap_explainer import SHAPExplanation
from integrated_explainer import IntegratedExplanation


@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    figsize: Tuple[int, int] = (16, 12)
    color_positive: str = '#2E8B57'  # Sea Green
    color_negative: str = '#DC143C'  # Crimson
    color_neutral: str = '#4682B4'   # Steel Blue
    font_size: int = 10
    title_size: int = 14
    label_size: int = 9
    dpi: int = 300
    bar_height: float = 0.6
    spacing: float = 0.1


class AdvancedExplanationVisualizer:
    """Advanced visualization tools for explanations"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize visualizer with configuration"""
        self.config = config or VisualizationConfig()
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_lime_explanation(self, 
                            explanation: LIMEExplanation,
                            top_n: int = 15,
                            show_values: bool = True,
                            save_path: Optional[str] = None) -> None:
        """Create comprehensive LIME explanation visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Feature importance bar chart
        self._plot_lime_feature_importance(ax1, explanation, top_n, show_values)
        
        # Plot 2: Prediction breakdown
        self._plot_lime_prediction_breakdown(ax2, explanation)
        
        plt.suptitle('LIME Explanation Analysis', fontsize=self.config.title_size, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.show()
    
    def _plot_lime_feature_importance(self, 
                                    ax: plt.Axes,
                                    explanation: LIMEExplanation,
                                    top_n: int,
                                    show_values: bool) -> None:
        """Plot LIME feature importance with clear, non-overlapping labels"""
        # Clear the axis first to prevent label bleeding
        ax.clear()
        
        # Get top N features
        sorted_features = sorted(explanation.feature_importance.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        if not sorted_features:
            ax.text(0.5, 0.5, 'No features to display', ha='center', va='center', transform=ax.transAxes)
            return
        
        features, values = zip(*sorted_features)
        
        # Create horizontal bar chart with optimal spacing
        y_pos = np.arange(len(features)) * 0.8
        colors = [self.config.color_positive if v > 0 else self.config.color_negative for v in values]
        
        bars = ax.barh(y_pos, values, height=0.6, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=0.2)
        
        # Calculate proper axis limits with extra space for labels
        x_min = min(values) * 1.4 if min(values) < 0 else min(values) * 0.6
        x_max = max(values) * 1.4 if max(values) > 0 else max(values) * 0.6
        
        # Add value labels with proper positioning and spacing
        if show_values:
            for i, (bar, value) in enumerate(zip(bars, values)):
                if abs(value) > 0.001:  # Only show labels for significant values
                    width = bar.get_width()
                    # Position labels well outside the bars to prevent overlap
                    if width > 0:  # Positive values - label on the right
                        label_x = x_max * 0.95  # Position at 95% of max x
                        ha = 'right'
                    else:  # Negative values - label on the left
                        label_x = x_min * 0.95  # Position at 95% of min x
                        ha = 'left'
                    
                    # Add background box to make labels more readable
                    ax.text(label_x, bar.get_y() + bar.get_height()/2,
                           f'{value:.3f}', ha=ha, va='center', 
                           fontsize=7, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                   alpha=0.9, edgecolor='gray', linewidth=0.5))
        
        # Format labels with proper truncation
        ax.set_yticks(y_pos)
        formatted_features = []
        for f in features:
            formatted = f.replace('_', ' ').title()
            if len(formatted) > 12:
                formatted = formatted[:9] + '...'
            formatted_features.append(formatted)
        
        ax.set_yticklabels(formatted_features, fontsize=8)
        ax.set_xlabel('Feature Importance', fontsize=9, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
        
        # Set axis limits with proper margins
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, len(features) - 0.5)
        
        # Adjust layout to prevent overlap
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
    
    def _plot_lime_prediction_breakdown(self, ax: plt.Axes, explanation: LIMEExplanation) -> None:
        """Plot LIME prediction breakdown"""
        # Calculate positive and negative contributions
        positive_contrib = sum(v for v in explanation.feature_importance.values() if v > 0)
        negative_contrib = sum(v for v in explanation.feature_importance.values() if v < 0)
        
        # Create pie chart
        labels = ['Positive\nContributions', 'Negative\nContributions']
        sizes = [positive_contrib, abs(negative_contrib)]
        colors = [self.config.color_positive, self.config.color_negative]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90, textprops={'fontsize': 10})
        
        # Add prediction value in center
        ax.text(0, 0, f'Prediction:\n{explanation.prediction:.3f}', ha='center', va='center',
               fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_title('Prediction Breakdown', fontsize=self.config.font_size, fontweight='bold')
    
    def plot_shap_explanation(self, 
                            explanation: SHAPExplanation,
                            top_n: int = 15,
                            show_values: bool = True,
                            save_path: Optional[str] = None) -> None:
        """Create comprehensive SHAP explanation visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Waterfall chart
        self._plot_shap_waterfall(ax1, explanation, top_n, show_values)
        
        # Plot 2: Feature importance
        self._plot_shap_importance(ax2, explanation, top_n, show_values)
        
        # Plot 3: Prediction flow
        self._plot_shap_prediction_flow(ax3, explanation)
        
        # Plot 4: Feature distribution
        self._plot_shap_feature_distribution(ax4, explanation)
        
        plt.suptitle('SHAP Explanation Analysis', fontsize=self.config.title_size, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.show()
    
    def _plot_shap_waterfall(self, 
                           ax: plt.Axes,
                           explanation: SHAPExplanation,
                           top_n: int,
                           show_values: bool) -> None:
        """Plot SHAP waterfall chart"""
        # Get top N features
        sorted_features = sorted(explanation.shap_values.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        if not sorted_features:
            ax.text(0.5, 0.5, 'No features to display', ha='center', va='center', transform=ax.transAxes)
            return
        
        features, values = zip(*sorted_features)
        
        # Create horizontal bar chart with better spacing
        y_pos = np.arange(len(features)) * (1 + self.config.spacing)
        colors = [self.config.color_positive if v > 0 else self.config.color_negative for v in values]
        
        bars = ax.barh(y_pos, values, height=self.config.bar_height, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=0.3)
        
        # Add value labels with better positioning
        if show_values:
            for i, (bar, value) in enumerate(zip(bars, values)):
                width = bar.get_width()
                label_x = width + (0.005 if width > 0 else -0.005)
                ax.text(label_x, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='left' if width > 0 else 'right', va='center', 
                       fontsize=self.config.label_size, fontweight='bold')
        
        # Add base value and prediction lines
        ax.axvline(explanation.base_value, color='black', linestyle='--', alpha=0.8, 
                  linewidth=2, label=f'Base Value ({explanation.base_value:.3f})')
        ax.axvline(explanation.prediction, color='blue', linestyle='-', alpha=0.8, 
                  linewidth=2, label=f'Prediction ({explanation.prediction:.3f})')
        
        # Formatting with better label handling
        ax.set_yticks(y_pos)
        # Truncate long feature names
        formatted_features = []
        for f in features:
            formatted = f.replace('_', ' ').title()
            if len(formatted) > 20:
                formatted = formatted[:17] + '...'
            formatted_features.append(formatted)
        
        ax.set_yticklabels(formatted_features, fontsize=self.config.label_size)
        ax.set_xlabel('SHAP Value', fontsize=self.config.font_size, fontweight='bold')
        ax.set_title('SHAP Waterfall Plot', fontsize=self.config.title_size, fontweight='bold', pad=20)
        ax.legend(fontsize=self.config.label_size, loc='upper right')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Adjust layout
        ax.tick_params(axis='y', labelsize=self.config.label_size)
        ax.tick_params(axis='x', labelsize=self.config.label_size)
    
    def _plot_shap_importance(self, 
                            ax: plt.Axes,
                            explanation: SHAPExplanation,
                            top_n: int,
                            show_values: bool) -> None:
        """Plot SHAP feature importance with clear, non-overlapping labels"""
        # Clear the axis first to prevent label bleeding
        ax.clear()
        
        # Get top N features by absolute SHAP value
        sorted_features = sorted(explanation.shap_values.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        if not sorted_features:
            ax.text(0.5, 0.5, 'No features to display', ha='center', va='center', transform=ax.transAxes)
            return
        
        features, values = zip(*sorted_features)
        
        # Create horizontal bar chart with optimal spacing
        y_pos = np.arange(len(features)) * 0.8
        bars = ax.barh(y_pos, np.abs(values), color=self.config.color_neutral, alpha=0.7, 
                      edgecolor='black', linewidth=0.2, height=0.6)
        
        # Calculate proper axis limits with extra space for labels
        max_val = max(np.abs(values)) if values else 1
        x_max = max_val * 1.4
        
        # Add value labels with proper positioning and spacing
        if show_values:
            for i, (bar, value) in enumerate(zip(bars, values)):
                if abs(value) > 0.001:  # Only show labels for significant values
                    # Position labels well outside the bars to prevent overlap
                    label_x = x_max * 0.95  # Position at 95% of max x
                    
                    # Add background box to make labels more readable
                    ax.text(label_x, bar.get_y() + bar.get_height()/2,
                           f'{value:.3f}', ha='right', va='center', 
                           fontsize=7, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                   alpha=0.9, edgecolor='gray', linewidth=0.5))
        
        # Format labels with proper truncation
        ax.set_yticks(y_pos)
        formatted_features = []
        for f in features:
            formatted = f.replace('_', ' ').title()
            if len(formatted) > 12:
                formatted = formatted[:9] + '...'
            formatted_features.append(formatted)
        
        ax.set_yticklabels(formatted_features, fontsize=8)
        ax.set_xlabel('|SHAP Value|', fontsize=9, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Set axis limits with proper margins
        ax.set_xlim(0, x_max)
        ax.set_ylim(-0.5, len(features) - 0.5)
        
        # Adjust layout
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
    
    def _plot_shap_prediction_flow(self, ax: plt.Axes, explanation: SHAPExplanation) -> None:
        """Plot SHAP prediction flow"""
        # Calculate contributions
        positive_contrib = sum(v for v in explanation.shap_values.values() if v > 0)
        negative_contrib = sum(v for v in explanation.shap_values.values() if v < 0)
        
        # Create flow diagram
        x_positions = [0, 1, 2, 3]
        y_positions = [0, 0, 0, 0]
        
        # Base value
        ax.scatter(x_positions[0], y_positions[0], s=200, c=self.config.color_neutral, alpha=0.8, zorder=3)
        ax.text(x_positions[0], y_positions[0] + 0.1, f'Base\n{explanation.base_value:.3f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Positive contributions
        ax.scatter(x_positions[1], y_positions[1], s=200, c=self.config.color_positive, alpha=0.8, zorder=3)
        ax.text(x_positions[1], y_positions[1] + 0.1, f'Positive\n+{positive_contrib:.3f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Negative contributions
        ax.scatter(x_positions[2], y_positions[2], s=200, c=self.config.color_negative, alpha=0.8, zorder=3)
        ax.text(x_positions[2], y_positions[2] + 0.1, f'Negative\n{negative_contrib:.3f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Final prediction
        ax.scatter(x_positions[3], y_positions[3], s=200, c='blue', alpha=0.8, zorder=3)
        ax.text(x_positions[3], y_positions[3] + 0.1, f'Prediction\n{explanation.prediction:.3f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Draw arrows
        for i in range(len(x_positions) - 1):
            ax.annotate('', xy=(x_positions[i+1], y_positions[i+1]), xytext=(x_positions[i], y_positions[i]),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.7))
        
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.3, 0.3)
        ax.set_title('Prediction Flow', fontsize=self.config.font_size, fontweight='bold')
        ax.axis('off')
    
    def _plot_shap_feature_distribution(self, ax: plt.Axes, explanation: SHAPExplanation) -> None:
        """Plot SHAP feature value distribution"""
        # Get SHAP values
        values = list(explanation.shap_values.values())
        
        if not values:
            ax.text(0.5, 0.5, 'No features to display', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create histogram
        ax.hist(values, bins=20, alpha=0.7, color=self.config.color_neutral, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero Line')
        ax.axvline(np.mean(values), color='blue', linestyle='-', alpha=0.7, label=f'Mean: {np.mean(values):.3f}')
        
        ax.set_xlabel('SHAP Value', fontsize=self.config.font_size)
        ax.set_ylabel('Frequency', fontsize=self.config.font_size)
        ax.set_title('SHAP Value Distribution', fontsize=self.config.font_size, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    def plot_integrated_explanation(self, 
                                  explanation: IntegratedExplanation,
                                  save_path: Optional[str] = None) -> None:
        """Create simplified 4-chart explanation visualization with complete isolation"""
        # Use optimal figure size for 4 charts
        fig = plt.figure(figsize=(18, 12))
        
        # Create a 2x2 grid with very generous spacing
        gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.5)
        
        # Title
        fig.suptitle('Integrated Learning Plan Explanation', fontsize=16, fontweight='bold', y=0.95)
        
        # Plot 1: LIME Analysis (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('LIME Analysis', fontsize=14, fontweight='bold', pad=20)
        if explanation.lime_explanation:
            self._plot_lime_feature_importance(ax1, explanation.lime_explanation, 6, True)
        else:
            ax1.text(0.5, 0.5, 'LIME Analysis\nNot Available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12, style='italic')
            ax1.axis('off')
        
        # Plot 2: SHAP Analysis (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('SHAP Analysis', fontsize=14, fontweight='bold', pad=20)
        if explanation.shap_explanation:
            self._plot_shap_importance(ax2, explanation.shap_explanation, 6, True)
        else:
            ax2.text(0.5, 0.5, 'SHAP Analysis\nNot Available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12, style='italic')
            ax2.axis('off')
        
        # Plot 3: Agent Confidence (bottom-left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_title('Agent Confidence', fontsize=14, fontweight='bold', pad=20)
        self._plot_agent_confidence(ax3, explanation.agent_explanations)
        
        # Plot 4: Prediction Summary (bottom-right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title('Prediction Summary', fontsize=14, fontweight='bold', pad=20)
        self._plot_prediction_summary(ax4, explanation)
        
        # Ensure tight layout to prevent overlap
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.show()
    
    def _plot_learning_plan_overview(self, ax: plt.Axes, explanation: IntegratedExplanation) -> None:
        """Plot learning plan overview"""
        plan = explanation.learning_plan
        
        # Create overview text with better formatting
        overview_text = f"""Topic: {plan.user_input.topic}
Background: {plan.user_input.background[:80]}...
Format: {plan.user_input.preferred_format.value}
Prediction: {explanation.prediction:.3f}
Confidence: {explanation.confidence:.3f}"""
        
        ax.text(0.05, 0.5, overview_text, transform=ax.transAxes, fontsize=self.config.font_size, 
               verticalalignment='center', fontweight='normal',
               bbox=dict(boxstyle='round,pad=1.0', facecolor='lightblue', alpha=0.8, 
                        edgecolor='navy', linewidth=1))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Learning Plan Overview', fontsize=self.config.title_size, fontweight='bold', pad=20)
    
    def _plot_global_importance(self, ax: plt.Axes, global_importance: Dict[str, float], top_n: int) -> None:
        """Plot global feature importance"""
        if not global_importance:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Get top N features
        top_features = list(global_importance.items())[:top_n]
        features, importances = zip(*top_features)
        
        # Create horizontal bar chart with better spacing
        y_pos = np.arange(len(features)) * (1 + self.config.spacing)
        bars = ax.barh(y_pos, importances, height=self.config.bar_height, 
                      color=self.config.color_neutral, alpha=0.8, edgecolor='black', linewidth=0.3)
        
        # Add value labels with better positioning
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center', 
                   fontsize=self.config.label_size, fontweight='bold')
        
        # Format labels with truncation
        ax.set_yticks(y_pos)
        formatted_features = []
        for f in features:
            formatted = f.replace('_', ' ').title()
            if len(formatted) > 20:
                formatted = formatted[:17] + '...'
            formatted_features.append(formatted)
        
        ax.set_yticklabels(formatted_features, fontsize=self.config.label_size)
        ax.set_xlabel('Importance', fontsize=self.config.font_size, fontweight='bold')
        ax.set_title('Global Feature Importance', fontsize=self.config.title_size, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Adjust layout
        ax.tick_params(axis='y', labelsize=self.config.label_size)
        ax.tick_params(axis='x', labelsize=self.config.label_size)
    
    def _plot_agent_confidence(self, ax: plt.Axes, agent_explanations: List[Any]) -> None:
        """Plot agent confidence levels with clear labels"""
        # Clear the axis first to prevent label bleeding
        ax.clear()
        
        if not agent_explanations:
            ax.text(0.5, 0.5, 'No agent data', ha='center', va='center', transform=ax.transAxes)
            return
        
        agents = [exp.agent_name for exp in agent_explanations]
        confidences = [exp.confidence_score for exp in agent_explanations]
        
        # Create bars with optimal spacing
        x_pos = np.arange(len(agents))
        bars = ax.bar(x_pos, confidences, color=self.config.color_neutral, alpha=0.8, 
                     edgecolor='black', linewidth=0.2, width=0.6)
        
        # Add value labels
        for bar, conf in zip(bars, confidences):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{conf:.2f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        # Format agent names with proper truncation
        formatted_agents = []
        for agent in agents:
            if len(agent) > 10:
                formatted_agents.append(agent[:7] + '...')
            else:
                formatted_agents.append(agent)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(formatted_agents, fontsize=9, rotation=45, ha='right')
        ax.set_ylabel('Confidence', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_xlim(-0.5, len(agents) - 0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Adjust layout
        ax.tick_params(axis='y', labelsize=9)
    
    def _plot_prediction_summary(self, ax: plt.Axes, explanation: IntegratedExplanation) -> None:
        """Plot prediction summary with clear labels"""
        # Clear the axis first to prevent label bleeding
        ax.clear()
        
        # Create summary metrics
        metrics = {
            'Prediction': explanation.prediction,
            'Confidence': explanation.confidence,
            'Base Value': explanation.shap_explanation.base_value if explanation.shap_explanation else 0,
            'Feature Count': len(explanation.global_feature_importance)
        }
        
        # Create bar chart with optimal spacing
        names = list(metrics.keys())
        values = list(metrics.values())
        x_pos = np.arange(len(names))
        
        bars = ax.bar(x_pos, values, color=self.config.color_neutral, alpha=0.8, 
                     edgecolor='black', linewidth=0.2, width=0.6)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.2f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=9, rotation=45, ha='right')
        ax.set_ylabel('Value', fontsize=10, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.2 if values else 1)
        ax.set_xlim(-0.5, len(names) - 0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Adjust layout
        ax.tick_params(axis='y', labelsize=9)
    
    def _plot_recommendations(self, ax: plt.Axes, explanation: IntegratedExplanation) -> None:
        """Plot recommendations"""
        # Extract recommendations from explanation summary
        summary_lines = explanation.explanation_summary.split('\n')
        recommendations = [line.strip() for line in summary_lines if line.strip().startswith(('•', '1.', '2.', '3.'))]
        
        if not recommendations:
            ax.text(0.5, 0.5, 'No recommendations', ha='center', va='center', transform=ax.transAxes,
                   fontsize=self.config.font_size, style='italic')
            return
        
        # Display recommendations as text with better formatting
        rec_text = '\n'.join(recommendations[:6])  # Top 6 recommendations
        ax.text(0.05, 0.95, rec_text, transform=ax.transAxes, fontsize=self.config.label_size, 
               verticalalignment='top', fontweight='normal',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8, 
                        edgecolor='orange', linewidth=1))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Recommendations', fontsize=self.config.title_size, fontweight='bold', pad=20)


def demonstrate_visualizations():
    """Demonstrate visualization capabilities"""
    print("ADVANCED EXPLANATION VISUALIZATION DEMO")
    print("=" * 70)
    
    # Create sample data
    from integrated_explainer import create_integrated_system
    from feature_extractor import create_sample_learning_plans
    system = create_integrated_system()
    learning_plans = create_sample_learning_plans()
    
    # Create visualizer
    visualizer = AdvancedExplanationVisualizer()
    
    # Generate explanations
    explanation = system.explain_learning_plan(learning_plans[0], [])
    
    # Create visualizations
    if explanation.lime_explanation:
        print("Creating LIME visualization...")
        visualizer.plot_lime_explanation(explanation.lime_explanation)
    
    if explanation.shap_explanation:
        print("Creating SHAP visualization...")
        visualizer.plot_shap_explanation(explanation.shap_explanation)
    
    print("Creating integrated visualization...")
    visualizer.plot_integrated_explanation(explanation)


if __name__ == "__main__":
    demonstrate_visualizations()
