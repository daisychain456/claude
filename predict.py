#!/usr/bin/env python3
"""
Make predictions on new data using the trained language retention model.

This script:
1. Loads a trained model
2. Loads new data
3. Makes predictions
4. Saves results with probability scores
"""

import numpy as np
import pandas as pd
import argparse
import pickle
import os


def load_model(model_path='models/language_retention_model.pkl'):
    """
    Load trained model and preprocessing pipeline.

    Parameters:
    -----------
    model_path : str
        Path to saved model

    Returns:
    --------
    dict
        Model data including model, scaler, and feature names
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def preprocess_new_data(df, model_data):
    """
    Preprocess new data for prediction.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    model_data : dict
        Loaded model data

    Returns:
    --------
    X : np.ndarray
        Preprocessed feature matrix
    """
    # Extract features in the same order as training
    feature_cols = model_data['feature_names']

    # Check for missing columns
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract features
    X = df[feature_cols].values

    # Handle missing values (fill with training medians)
    medians = model_data['medians']
    for i in range(X.shape[1]):
        X[np.isnan(X[:, i]), i] = medians[i]

    # Scale using training scaler
    scaler = model_data['scaler']
    X_scaled = scaler.transform(X)

    return X_scaled


def make_predictions(df, model_data):
    """
    Make predictions on new data.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    model_data : dict
        Loaded model data

    Returns:
    --------
    pd.DataFrame
        Original data with predictions and probabilities
    """
    # Preprocess data
    X = preprocess_new_data(df, model_data)

    # Get model
    model = model_data['best_model']

    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Create results dataframe
    results = df.copy()
    results['predicted_speaks_parent_language'] = predictions
    results['probability_no'] = probabilities[:, 0]
    results['probability_yes'] = probabilities[:, 1]
    results['confidence'] = np.max(probabilities, axis=1)

    # Add interpretation
    def interpret_prediction(row):
        if row['predicted_speaks_parent_language'] == 1:
            return f"Likely to speak (confidence: {row['confidence']:.1%})"
        else:
            return f"Likely not to speak (confidence: {row['confidence']:.1%})"

    results['interpretation'] = results.apply(interpret_prediction, axis=1)

    return results


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(description='Make predictions on new data')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file with new data')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions (default: input_predictions.csv)')
    parser.add_argument('--model', type=str, default='models/language_retention_model.pkl',
                        help='Path to trained model')

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_predictions.csv"

    print("="*60)
    print("LANGUAGE RETENTION PREDICTION")
    print("="*60)

    # Load model
    print(f"\nLoading model from {args.model}...")
    model_data = load_model(args.model)
    print(f"✓ Loaded model: {model_data['best_model_name']}")
    print(f"  Trained on: {model_data['trained_date']}")

    # Load new data
    print(f"\nLoading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"✓ Loaded {len(df)} samples")

    # Make predictions
    print("\nMaking predictions...")
    results = make_predictions(df, model_data)
    print(f"✓ Predictions complete")

    # Summary statistics
    n_speak = (results['predicted_speaks_parent_language'] == 1).sum()
    n_not_speak = (results['predicted_speaks_parent_language'] == 0).sum()
    avg_confidence = results['confidence'].mean()

    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"\nTotal predictions: {len(results)}")
    print(f"Predicted to speak parent language: {n_speak} ({n_speak/len(results)*100:.1f}%)")
    print(f"Predicted NOT to speak parent language: {n_not_speak} ({n_not_speak/len(results)*100:.1f}%)")
    print(f"Average confidence: {avg_confidence:.1%}")

    # Save results
    results.to_csv(args.output, index=False)
    print(f"\n✓ Predictions saved to: {args.output}")

    # Show first few predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (first 5 rows)")
    print("="*60)
    display_cols = ['child_id', 'predicted_speaks_parent_language', 'probability_yes', 'confidence', 'interpretation']
    available_cols = [col for col in display_cols if col in results.columns]
    print(results[available_cols].head().to_string(index=False))

    print("\n" + "="*60)
    print("PREDICTION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
