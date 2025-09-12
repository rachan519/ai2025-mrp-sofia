"""
Feature Extraction System for Learning Plan Components
Extracts numerical and categorical features from learning plan data for LIME/SHAP analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re
from dataclasses import dataclass

from models import UserInput, KnowledgeGap, TopicPlan, TopicDetail, CompleteLearningPlan


@dataclass
class FeatureVector:
    """Container for extracted features"""
    numerical_features: np.ndarray
    categorical_features: np.ndarray
    text_features: np.ndarray
    feature_names: List[str]
    categorical_labels: Dict[str, List[str]]


class LearningPlanFeatureExtractor:
    """Extracts features from learning plan components for model-agnostic explainability"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_features(self, learning_plan: CompleteLearningPlan) -> FeatureVector:
        """Extract comprehensive features from a complete learning plan"""
        
        # Extract features from each component
        user_features = self._extract_user_features(learning_plan.user_input)
        gap_features = self._extract_gap_features(learning_plan.knowledge_gap)
        plan_features = self._extract_plan_features(learning_plan.topic_plan)
        detail_features = self._extract_detail_features(learning_plan.topic_details)
        
        # Combine all features
        all_features = {
            **user_features,
            **gap_features,
            **plan_features,
            **detail_features
        }
        
        # Separate numerical, categorical, and text features
        numerical_features, categorical_features, text_features, feature_names = self._categorize_features(all_features)
        
        return FeatureVector(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            text_features=text_features,
            feature_names=feature_names,
            categorical_labels=self._get_categorical_labels()
        )
    
    def _extract_user_features(self, user_input: UserInput) -> Dict[str, Any]:
        """Extract features from user input"""
        features = {}
        
        # Topic complexity features
        features['topic_length'] = len(user_input.topic.split())
        features['topic_has_numbers'] = bool(re.search(r'\d', user_input.topic))
        features['topic_has_special_chars'] = bool(re.search(r'[^a-zA-Z0-9\s]', user_input.topic))
        
        # Background features
        features['background_length'] = len(user_input.background.split())
        features['background_has_experience'] = any(word in user_input.background.lower() 
                                                  for word in ['experience', 'worked', 'used', 'familiar'])
        features['background_has_education'] = any(word in user_input.background.lower() 
                                                 for word in ['degree', 'studied', 'course', 'university', 'college'])
        features['background_has_programming'] = any(word in user_input.background.lower() 
                                                   for word in ['programming', 'coding', 'python', 'java', 'javascript'])
        
        # Learning format features
        features['preferred_format_video'] = 1 if user_input.preferred_format.value == 'video' else 0
        features['preferred_format_text'] = 1 if user_input.preferred_format.value == 'text' else 0
        features['preferred_format_audio'] = 1 if user_input.preferred_format.value == 'audio' else 0
        
        # Text features
        features['topic_text'] = user_input.topic
        features['background_text'] = user_input.background
        
        return features
    
    def _extract_gap_features(self, knowledge_gap: KnowledgeGap) -> Dict[str, Any]:
        """Extract features from knowledge gap analysis"""
        features = {}
        
        # Gap count features
        features['num_gaps'] = len(knowledge_gap.identified_gaps)
        features['avg_gap_length'] = np.mean([len(gap.split()) for gap in knowledge_gap.identified_gaps]) if knowledge_gap.identified_gaps else 0
        
        # Level features
        features['current_level_beginner'] = 1 if 'beginner' in knowledge_gap.current_level.lower() else 0
        features['current_level_intermediate'] = 1 if 'intermediate' in knowledge_gap.current_level.lower() else 0
        features['current_level_advanced'] = 1 if 'advanced' in knowledge_gap.current_level.lower() else 0
        
        features['target_level_beginner'] = 1 if 'beginner' in knowledge_gap.target_level.lower() else 0
        features['target_level_intermediate'] = 1 if 'intermediate' in knowledge_gap.target_level.lower() else 0
        features['target_level_advanced'] = 1 if 'advanced' in knowledge_gap.target_level.lower() else 0
        
        # Gap analysis features
        features['gap_analysis_length'] = len(knowledge_gap.gap_analysis.split())
        features['gap_analysis_has_technical'] = any(word in knowledge_gap.gap_analysis.lower() 
                                                   for word in ['technical', 'programming', 'coding', 'algorithm'])
        
        # Text features
        features['gaps_text'] = ' '.join(knowledge_gap.identified_gaps)
        features['gap_analysis_text'] = knowledge_gap.gap_analysis
        
        return features
    
    def _extract_plan_features(self, topic_plan: TopicPlan) -> Dict[str, Any]:
        """Extract features from topic plan"""
        features = {}
        
        # Plan structure features
        features['num_main_topics'] = len(topic_plan.main_topics)
        features['num_subtopics'] = len(topic_plan.subtopics)
        features['num_objectives'] = len(topic_plan.learning_objectives)
        
        # Duration features
        duration_text = topic_plan.estimated_duration.lower()
        features['duration_weeks'] = self._extract_duration_weeks(duration_text)
        features['duration_hours_per_week'] = self._extract_hours_per_week(duration_text)
        
        # Topic complexity features
        features['avg_topic_length'] = np.mean([len(topic.split()) for topic in topic_plan.main_topics]) if topic_plan.main_topics else 0
        features['avg_objective_length'] = np.mean([len(obj.split()) for obj in topic_plan.learning_objectives]) if topic_plan.learning_objectives else 0
        
        # Text features
        features['main_topics_text'] = ' '.join(topic_plan.main_topics)
        features['objectives_text'] = ' '.join(topic_plan.learning_objectives)
        
        return features
    
    def _extract_detail_features(self, topic_details: List[TopicDetail]) -> Dict[str, Any]:
        """Extract features from topic details"""
        features = {}
        
        if not topic_details:
            features['num_details'] = 0
            features['avg_resources_per_topic'] = 0
            features['avg_exercises_per_topic'] = 0
            features['avg_description_length'] = 0
            features['details_text'] = ''
            return features
        
        # Detail count features
        features['num_details'] = len(topic_details)
        
        # Resource and exercise features
        all_resources = []
        all_exercises = []
        all_descriptions = []
        
        for detail in topic_details:
            all_resources.extend(detail.resources)
            all_exercises.extend(detail.exercises)
            all_descriptions.append(detail.description)
        
        features['avg_resources_per_topic'] = len(all_resources) / len(topic_details)
        features['avg_exercises_per_topic'] = len(all_exercises) / len(topic_details)
        features['avg_description_length'] = np.mean([len(desc.split()) for desc in all_descriptions])
        
        # Content type features
        features['has_video_resources'] = any('video' in resource.lower() for resource in all_resources)
        features['has_text_resources'] = any('book' in resource.lower() or 'article' in resource.lower() for resource in all_resources)
        features['has_practical_exercises'] = any('practical' in exercise.lower() or 'project' in exercise.lower() for exercise in all_exercises)
        
        # Text features
        features['details_text'] = ' '.join(all_descriptions)
        features['resources_text'] = ' '.join(all_resources)
        features['exercises_text'] = ' '.join(all_exercises)
        
        return features
    
    def _categorize_features(self, all_features: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Categorize features into numerical, categorical, and text"""
        numerical_features = []
        categorical_features = []
        text_features = []
        feature_names = []
        
        for name, value in all_features.items():
            if isinstance(value, str) and name.endswith('_text'):
                # Text features - skip for now, focus on numerical/categorical
                continue
            elif isinstance(value, (int, float)) and not name.endswith('_text'):
                # Numerical features
                numerical_features.append(value)
                feature_names.append(name)
            elif isinstance(value, (int, float)) and value in [0, 1]:
                # Binary categorical features
                categorical_features.append(value)
                feature_names.append(name)
        
        return (np.array(numerical_features), 
                np.array(categorical_features), 
                np.array(text_features), 
                feature_names)
    
    def _extract_duration_weeks(self, duration_text: str) -> float:
        """Extract number of weeks from duration text"""
        # Look for patterns like "4-6 weeks", "2 weeks", "1 month"
        week_patterns = [
            r'(\d+)-(\d+)\s*weeks?',
            r'(\d+)\s*weeks?',
            r'(\d+)\s*month',
        ]
        
        for pattern in week_patterns:
            match = re.search(pattern, duration_text)
            if match:
                if '-' in match.group(0):
                    return (int(match.group(1)) + int(match.group(2))) / 2
                else:
                    return float(match.group(1))
        
        return 4.0  # Default fallback
    
    def _extract_hours_per_week(self, duration_text: str) -> float:
        """Extract hours per week from duration text"""
        # Look for patterns like "2-3 hours per week", "1 hour per week"
        hour_patterns = [
            r'(\d+)-(\d+)\s*hours?\s*per\s*week',
            r'(\d+)\s*hours?\s*per\s*week',
        ]
        
        for pattern in hour_patterns:
            match = re.search(pattern, duration_text)
            if match:
                if '-' in match.group(0):
                    return (int(match.group(1)) + int(match.group(2))) / 2
                else:
                    return float(match.group(1))
        
        return 2.5  # Default fallback
    
    def _get_categorical_labels(self) -> Dict[str, List[str]]:
        """Get labels for categorical features"""
        return {
            'preferred_format': ['video', 'text', 'audio'],
            'current_level': ['beginner', 'intermediate', 'advanced'],
            'target_level': ['beginner', 'intermediate', 'advanced']
        }
    
    def fit_transform(self, learning_plans: List[CompleteLearningPlan]) -> Tuple[np.ndarray, List[str]]:
        """Fit the feature extractor and transform learning plans to feature matrix"""
        # Extract features for all learning plans
        feature_vectors = [self.extract_features(plan) for plan in learning_plans]
        
        # Combine numerical and categorical features
        combined_features = []
        feature_names = []
        
        for fv in feature_vectors:
            combined = np.concatenate([fv.numerical_features, fv.categorical_features])
            combined_features.append(combined)
            if not feature_names:
                feature_names = fv.feature_names
        
        # Fit scaler
        feature_matrix = np.array(combined_features)
        self.scaler.fit(feature_matrix)
        scaled_features = self.scaler.transform(feature_matrix)
        
        self.is_fitted = True
        return scaled_features, feature_names
    
    def transform(self, learning_plan: CompleteLearningPlan) -> np.ndarray:
        """Transform a single learning plan to feature vector"""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        fv = self.extract_features(learning_plan)
        combined = np.concatenate([fv.numerical_features, fv.categorical_features])
        return self.scaler.transform(combined.reshape(1, -1)).flatten()


class LearningPlanPredictor:
    """Simple predictor for learning plan quality/effectiveness"""
    
    def __init__(self):
        self.feature_weights = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the predictor with feature matrix and target values"""
        # Add regularization to avoid singular matrix issues
        regularization = 0.01 * np.eye(X.shape[1])
        self.feature_weights = np.linalg.pinv(X.T @ X + regularization) @ X.T @ y
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values"""
        if not self.is_fitted:
            raise ValueError("Predictor must be fitted before predict")
        
        return X @ self.feature_weights
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification)"""
        predictions = self.predict(X)
        # Convert to probabilities using sigmoid
        return 1 / (1 + np.exp(-predictions))


def create_sample_learning_plans() -> List[CompleteLearningPlan]:
    """Create sample learning plans for testing"""
    from models import LearningFormat
    
    plans = []
    
    # Sample 1: Beginner Python
    user_input1 = UserInput(
        topic="Python Programming",
        background="Complete beginner with no programming experience",
        preferred_format=LearningFormat.VIDEO
    )
    
    knowledge_gap1 = KnowledgeGap(
        identified_gaps=["Basic programming concepts", "Python syntax", "Problem solving"],
        current_level="Complete beginner",
        target_level="Intermediate",
        gap_analysis="Need to learn fundamental programming concepts and Python syntax"
    )
    
    topic_plan1 = TopicPlan(
        main_topics=["Introduction to Programming", "Python Basics", "Data Structures"],
        subtopics=["Variables", "Functions", "Lists"],
        learning_objectives=["Understand basic concepts", "Write simple programs", "Use data structures"],
        estimated_duration="6-8 weeks with 3-4 hours per week"
    )
    
    topic_details1 = [
        TopicDetail(
            topic_name="Introduction to Programming",
            description="Learn fundamental programming concepts",
            resources=["Video tutorials", "Interactive exercises"],
            exercises=["Hello World", "Basic calculations"],
            assessment_criteria="Complete all exercises"
        )
    ]
    
    plan1 = CompleteLearningPlan(
        user_input=user_input1,
        knowledge_gap=knowledge_gap1,
        topic_plan=topic_plan1,
        topic_details=topic_details1,
        learning_path="Start with basics, practice regularly",
        recommended_resources=["Online courses", "Practice platforms"],
        timeline="6-8 weeks",
        success_metrics=["Complete exercises", "Build projects"]
    )
    
    plans.append(plan1)
    
    # Sample 2: Advanced Machine Learning
    user_input2 = UserInput(
        topic="Machine Learning",
        background="I have Python experience and know statistics",
        preferred_format=LearningFormat.TEXT
    )
    
    knowledge_gap2 = KnowledgeGap(
        identified_gaps=["ML algorithms", "Model evaluation", "Feature engineering"],
        current_level="Intermediate",
        target_level="Advanced",
        gap_analysis="Need to learn advanced ML techniques and best practices"
    )
    
    topic_plan2 = TopicPlan(
        main_topics=["Advanced Algorithms", "Model Evaluation", "Production Deployment"],
        subtopics=["Deep Learning", "Cross-validation", "MLOps"],
        learning_objectives=["Master advanced techniques", "Evaluate models properly", "Deploy to production"],
        estimated_duration="8-10 weeks with 5-6 hours per week"
    )
    
    topic_details2 = [
        TopicDetail(
            topic_name="Advanced Algorithms",
            description="Learn deep learning and ensemble methods",
            resources=["Research papers", "Advanced textbooks"],
            exercises=["Implement neural networks", "Build ensemble models"],
            assessment_criteria="Complete complex projects"
        )
    ]
    
    plan2 = CompleteLearningPlan(
        user_input=user_input2,
        knowledge_gap=knowledge_gap2,
        topic_plan=topic_plan2,
        topic_details=topic_details2,
        learning_path="Study theory, implement algorithms, build projects",
        recommended_resources=["Research papers", "Advanced courses"],
        timeline="8-10 weeks",
        success_metrics=["Complete projects", "Publish results"]
    )
    
    plans.append(plan2)
    
    return plans
