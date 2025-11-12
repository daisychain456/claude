#!/usr/bin/env python3
"""
Generate comprehensive visualizations for language retention prediction model.

This script creates:
1. Feature importance plot
2. Confusion matrix heatmap
3. ROC curve
4. Demographic pattern distributions
5. Correlation heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_model(model_path='models/language_retention_model.pkl'):
    """Load trained model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def plot_feature_importance(model_data, top_n=15, output_dir='results'):
    """
    Plot feature importance.

    Parameters:
    -----------
    model_data : dict
        Loaded model data
    top_n : int
        Number of top features to display
    output_dir : str
        Directory to save plot
    """
    if model_data['feature_importance'] is None:
        print("Feature importance not available for this model")
        return

    # Create dataframe
    feat_imp_df = pd.DataFrame({
        'Feature': model_data['feature_names'],
        'Importance': model_data['feature_importance']
    })
    feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False).head(top_n)

    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(feat_imp_df)), feat_imp_df['Importance'], color='steelblue')

    # Color the top 3 differently
    for i in range(min(3, len(bars))):
        bars[i].set_color('darkred')

    plt.yticks(range(len(feat_imp_df)), feat_imp_df['Feature'])
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Most Important Features for Language Retention\nModel: {model_data["best_model_name"]}',
              fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature importance plot: {output_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir='results'):
    """
    Plot confusion matrix heatmap.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    output_dir : str
        Directory to save plot
    """
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Does not speak', 'Speaks'],
                yticklabels=['Does not speak', 'Speaks'])

    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.7, f'({cm_pct[i, j]:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')

    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Language Retention Prediction',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix: {output_path}")
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, output_dir='results'):
    """
    Plot ROC curve.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    output_dir : str
        Directory to save plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkred', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve - Language Retention Prediction',
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curve: {output_path}")
    plt.close()


def plot_demographic_patterns(df, output_dir='results'):
    """
    Plot demographic patterns by language retention status.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with features and target
    output_dir : str
        Directory to save plot
    """
    # Select key features to visualize
    features_to_plot = [
        'parent1_english_proficiency',
        'distance_to_ethnic_enclave_km',
        'pct_same_ethnicity_neighborhood',
        'years_heritage_language_instruction',
        'hours_per_week_heritage_media',
        'pct_friends_same_ethnicity'
    ]

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]

        # Create violin plot
        data_to_plot = [
            df[df['speaks_parent_language'] == 0][feature].dropna(),
            df[df['speaks_parent_language'] == 1][feature].dropna()
        ]

        parts = ax.violinplot(data_to_plot, positions=[0, 1], showmeans=True, showmedians=True)

        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            if i == 0:
                pc.set_facecolor('lightcoral')
                pc.set_alpha(0.7)
            else:
                pc.set_facecolor('lightgreen')
                pc.set_alpha(0.7)

        # Format feature name for display
        feature_name = feature.replace('_', ' ').title()
        ax.set_title(feature_name, fontweight='bold', fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Does not speak', 'Speaks'], fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Distribution of Key Features by Language Retention Status',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'demographic_patterns.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved demographic patterns: {output_path}")
    plt.close()


def plot_correlation_heatmap(df, output_dir='results'):
    """
    Plot correlation heatmap of features.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with features
    output_dir : str
        Directory to save plot
    """
    # Select numeric columns (exclude ID)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'child_id']

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Create plot
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)

    plt.title('Feature Correlation Heatmap',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved correlation heatmap: {output_path}")
    plt.close()


def plot_retention_by_categories(df, output_dir='results'):
    """
    Plot retention rates by categorical variables.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with features and target
    output_dir : str
        Directory to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. By parents speaking heritage at home
    ax = axes[0, 0]
    retention_by_home = df.groupby('parents_speak_heritage_at_home')['speaks_parent_language'].mean() * 100
    bars = ax.bar([0, 1], retention_by_home, color=['lightcoral', 'lightgreen'], edgecolor='black')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'])
    ax.set_ylabel('Retention Rate (%)', fontweight='bold')
    ax.set_title('Retention by Parents Speaking Heritage at Home', fontweight='bold')
    ax.set_ylim([0, 100])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 2. By grandparents in household
    ax = axes[0, 1]
    retention_by_grandparents = df.groupby('grandparents_in_household')['speaks_parent_language'].mean() * 100
    bars = ax.bar([0, 1], retention_by_grandparents, color=['lightcoral', 'lightgreen'], edgecolor='black')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'])
    ax.set_ylabel('Retention Rate (%)', fontweight='bold')
    ax.set_title('Retention by Grandparents in Household', fontweight='bold')
    ax.set_ylim([0, 100])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 3. By heritage language school availability
    ax = axes[1, 0]
    retention_by_school = df.groupby('heritage_language_school_available')['speaks_parent_language'].mean() * 100
    bars = ax.bar([0, 1], retention_by_school, color=['lightcoral', 'lightgreen'], edgecolor='black')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'])
    ax.set_ylabel('Retention Rate (%)', fontweight='bold')
    ax.set_title('Retention by Heritage Language School Availability', fontweight='bold')
    ax.set_ylim([0, 100])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 4. By religious institution attendance
    ax = axes[1, 1]
    retention_by_religious = df.groupby('attends_religious_institution')['speaks_parent_language'].mean() * 100
    bars = ax.bar([0, 1], retention_by_religious, color=['lightcoral', 'lightgreen'], edgecolor='black')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'])
    ax.set_ylabel('Retention Rate (%)', fontweight='bold')
    ax.set_title('Retention by Religious Institution Attendance', fontweight='bold')
    ax.set_ylim([0, 100])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Language Retention Rates by Categorical Factors',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, 'retention_by_categories.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved retention by categories: {output_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    parser = argparse.ArgumentParser(description='Generate visualizations for language retention model')
    parser.add_argument('--data', type=str, default='data/language_retention_data.csv',
                        help='Path to data CSV')
    parser.add_argument('--model', type=str, default='models/language_retention_model.pkl',
                        help='Path to trained model')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save visualizations')

    args = parser.parse_args()

    print("="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.model}...")
    model_data = load_model(args.model)
    print(f"✓ Loaded model: {model_data['best_model_name']}")

    # Load data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"✓ Loaded {len(df)} samples")

    # Prepare data for evaluation
    feature_cols = model_data['feature_names']
    X = df[feature_cols].values
    y = df['speaks_parent_language'].values

    # Handle missing values
    medians = model_data['medians']
    for i in range(X.shape[1]):
        X[np.isnan(X[:, i]), i] = medians[i]

    # Scale
    X_scaled = model_data['scaler'].transform(X)

    # Split (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Get predictions
    model = model_data['best_model']
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60 + "\n")

    # Generate all plots
    plot_feature_importance(model_data, top_n=15, output_dir=args.output_dir)
    plot_confusion_matrix(y_test, y_pred, output_dir=args.output_dir)
    plot_roc_curve(y_test, y_pred_proba, output_dir=args.output_dir)
    plot_demographic_patterns(df, output_dir=args.output_dir)
    plot_correlation_heatmap(df, output_dir=args.output_dir)
    plot_retention_by_categories(df, output_dir=args.output_dir)

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nAll plots saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  - feature_importance.png")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - demographic_patterns.png")
    print("  - correlation_heatmap.png")
    print("  - retention_by_categories.png")


if __name__ == "__main__":
    main()
