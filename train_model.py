#!/usr/bin/env python3
"""
Train machine learning models to predict language retention in second-generation immigrant children.

This script:
1. Loads and preprocesses data
2. Trains multiple models (Logistic Regression, Random Forest, Gradient Boosting)
3. Evaluates model performance
4. Saves the best model and preprocessing pipeline
"""

import numpy as np
import pandas as pd
import argparse
import pickle
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


class LanguageRetentionModel:
    """
    Machine learning model for predicting language retention.
    """

    def __init__(self):
        """Initialize model components."""
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.feature_importance = None

    def load_data(self, filepath):
        """
        Load data from CSV file.

        Parameters:
        -----------
        filepath : str
            Path to CSV file

        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")
        return df

    def preprocess_data(self, df, fit_scaler=True):
        """
        Preprocess data: separate features and target, handle missing values, scale features.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fit_scaler : bool
            Whether to fit the scaler (True for training, False for prediction)

        Returns:
        --------
        X : np.ndarray
            Feature matrix
        y : np.ndarray or None
            Target vector (None if not present)
        """
        # Identify feature columns (exclude ID and target)
        exclude_cols = ['child_id', 'speaks_parent_language']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Store feature names
        if fit_scaler:
            self.feature_names = feature_cols

        # Extract features
        X = df[feature_cols].values

        # Handle missing values (fill with median)
        if fit_scaler:
            self.medians = np.nanmedian(X, axis=0)

        # Replace NaN with medians
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = self.medians[i]

        # Scale features
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        # Extract target if present
        y = None
        if 'speaks_parent_language' in df.columns:
            y = df['speaks_parent_language'].values

        print(f"✓ Preprocessed {X.shape[0]} samples with {X.shape[1]} features")

        return X, y

    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train multiple models and compare performance.

        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        """
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)

        # Define models
        model_configs = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=4
            )
        }

        results = []

        for name, model in model_configs.items():
            print(f"\nTraining {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Cross-validation on training set
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            # Store model and results
            self.models[name] = model
            results.append({
                'Model': name,
                'CV ROC-AUC': cv_scores.mean(),
                'CV Std': cv_scores.std(),
                'Test Accuracy': accuracy,
                'Test Precision': precision,
                'Test Recall': recall,
                'Test F1': f1,
                'Test ROC-AUC': roc_auc
            })

            print(f"  Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Test ROC-AUC: {roc_auc:.4f}")

        # Create results dataframe
        results_df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(results_df.to_string(index=False))

        # Select best model based on Test ROC-AUC
        best_idx = results_df['Test ROC-AUC'].idxmax()
        self.best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]

        print(f"\n✓ Best model: {self.best_model_name}")
        print(f"  ROC-AUC: {results_df.loc[best_idx, 'Test ROC-AUC']:.4f}")

        # Extract feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # For logistic regression, use absolute coefficients
            self.feature_importance = np.abs(self.best_model.coef_[0])
        else:
            self.feature_importance = None

        return results_df

    def evaluate_model(self, X_test, y_test):
        """
        Detailed evaluation of the best model.

        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        """
        print("\n" + "="*60)
        print(f"DETAILED EVALUATION - {self.best_model_name}")
        print("="*60)

        y_pred = self.best_model.predict(X_test)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Does not speak', 'Speaks']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"                 Predicted: No  |  Predicted: Yes")
        print(f"Actual: No       {cm[0,0]:8d}     |     {cm[0,1]:8d}")
        print(f"Actual: Yes      {cm[1,0]:8d}     |     {cm[1,1]:8d}")

        # Feature importance
        if self.feature_importance is not None:
            print("\n" + "="*60)
            print("TOP 10 MOST IMPORTANT FEATURES")
            print("="*60)

            # Create feature importance dataframe
            feat_imp_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.feature_importance
            })
            feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)

            for idx, row in feat_imp_df.head(10).iterrows():
                print(f"{row['Feature']:45s} {row['Importance']:.4f}")

    def save_model(self, filepath='models/language_retention_model.pkl'):
        """
        Save the trained model and preprocessing pipeline.

        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'medians': self.medians,
            'trained_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n✓ Model saved to: {filepath}")

    def load_model(self, filepath='models/language_retention_model.pkl'):
        """
        Load a trained model and preprocessing pipeline.

        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.medians = model_data['medians']

        print(f"✓ Model loaded: {self.best_model_name}")
        print(f"  Trained on: {model_data['trained_date']}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train language retention prediction model')
    parser.add_argument('--data', type=str, default='data/language_retention_data.csv',
                        help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='models/language_retention_model.pkl',
                        help='Path to save trained model')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data for testing')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    print("="*60)
    print("LANGUAGE RETENTION PREDICTION MODEL TRAINING")
    print("="*60)

    # Initialize model
    model = LanguageRetentionModel()

    # Load data
    df = model.load_data(args.data)

    # Preprocess data
    X, y = model.preprocess_data(df, fit_scaler=True)

    # Split data
    print(f"\nSplitting data (test size: {args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")

    # Train models
    results_df = model.train_models(X_train, y_train, X_test, y_test)

    # Evaluate best model
    model.evaluate_model(X_test, y_test)

    # Save model
    model.save_model(args.output)

    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/model_comparison.csv', index=False)
    print(f"✓ Results saved to: results/model_comparison.csv")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run visualize_results.py to see detailed visualizations")
    print("  2. Use predict.py to make predictions on new data")


if __name__ == "__main__":
    main()
