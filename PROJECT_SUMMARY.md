# Language Retention Predictor - Project Summary

## Overview

This project implements a machine learning system to predict whether second-generation immigrant children will retain fluency in their parents' native/heritage language. The system is designed to support research in sociolinguistics, education policy, and community language planning.

## Project Structure

```
language-retention-predictor/
├── data/
│   └── language_retention_data.csv      # Generated sample data (500 samples)
├── models/
│   └── language_retention_model.pkl     # Trained Random Forest model
├── results/
│   ├── model_comparison.csv             # Model performance comparison
│   ├── feature_importance.png           # Top 15 feature importance
│   ├── confusion_matrix.png             # Prediction accuracy matrix
│   ├── roc_curve.png                    # ROC curve (AUC = 0.855)
│   ├── demographic_patterns.png         # Feature distributions
│   ├── correlation_heatmap.png          # Feature correlations
│   └── retention_by_categories.png      # Retention rates by factors
│
├── data_template.csv                    # Template for data collection
├── generate_sample_data.py              # Generate synthetic training data
├── train_model.py                       # Train ML models
├── predict.py                           # Make predictions on new data
├── visualize_results.py                 # Generate all visualizations
├── requirements.txt                     # Python dependencies
├── README.md                            # Main documentation
├── USAGE_GUIDE.md                       # Detailed usage instructions
└── PROJECT_SUMMARY.md                   # This file
```

## Key Features

### 20 Predictive Variables

**Family Environment (9 features):**
- Parent English proficiency levels
- Years parents in host country
- Family size and composition
- Parents' education levels
- Home language use

**Community Context (3 features):**
- Distance to ethnic enclave
- Neighborhood ethnic composition
- Heritage language school availability

**Educational Exposure (4 features):**
- Years of heritage language instruction
- Religious institution attendance
- Heritage language in religious services
- Bilingual education participation

**Social & Cultural (4 features):**
- Friend group ethnic composition
- Heritage media consumption
- Visits to heritage country

### Machine Learning Models

Three models are trained and compared:
1. **Logistic Regression** - Baseline interpretable model
2. **Random Forest** - Best performing (93% accuracy, 0.855 ROC-AUC)
3. **Gradient Boosting** - Alternative ensemble method

The system automatically selects the best performing model based on ROC-AUC score.

### Visualization Suite

Six comprehensive visualizations:
1. Feature importance ranking
2. Confusion matrix with percentages
3. ROC curve with AUC score
4. Demographic pattern distributions
5. Feature correlation heatmap
6. Retention rates by categorical factors

## Model Performance

**On synthetic test data (100 samples):**
- **Accuracy**: 93%
- **Precision**: 94%
- **Recall**: 99%
- **F1-Score**: 96%
- **ROC-AUC**: 0.855

**Top 5 Most Important Features:**
1. Parents speak heritage language at home (14.9%)
2. Percentage same ethnicity in neighborhood (9.7%)
3. Parents' years in country (9.4%)
4. Distance to ethnic enclave (9.2%)
5. Child's age (8.4%)

## Key Findings from Sample Data

- **92.2%** of children in sample speak parent language
- **74.4%** of families speak heritage language at home
- Average distance to ethnic enclave: **7.5 km**
- Average parents' time in country: **14.8 years**

## Modularity & Extensibility

The system is designed for easy modification:

**Add New Features:**
- Simply add columns to your CSV
- Model automatically incorporates them
- Retrain to update predictions

**Retrain with New Data:**
```bash
python train_model.py --data new_data.csv
```

**Batch Predictions:**
```bash
python predict.py --input batch.csv --output predictions.csv
```

**Try Different Models:**
- Edit model_configs in train_model.py
- Add scikit-learn classifiers
- Compare performance automatically

## Use Cases

1. **Academic Research**
   - Study language retention patterns
   - Identify key protective factors
   - Inform theoretical models

2. **Education Policy**
   - Predict language program needs
   - Allocate bilingual education resources
   - Design intervention programs

3. **Community Planning**
   - Assess language vitality
   - Plan heritage language schools
   - Support cultural preservation

4. **Individual Assessment**
   - Predict individual child outcomes
   - Identify at-risk cases
   - Guide family language policy decisions

## Assumptions & Limitations

**Assumptions:**
1. Survey responses are accurate and honest
2. "Speaks parent language" defined as conversational fluency
3. Features remain relatively stable during childhood
4. Model generalizes across different heritage languages
5. Age range optimized for 5-18 years

**Limitations:**
1. Requires sufficient training data (300+ samples recommended)
2. May not capture all cultural nuances
3. Performance depends on data quality
4. Class imbalance may affect minority class predictions
5. Cannot capture dynamic/changing family situations

## Ethical Considerations

This tool should be used to:
- ✓ Support heritage language maintenance
- ✓ Understand community language patterns
- ✓ Inform supportive policies and programs
- ✓ Celebrate multilingualism

This tool should NOT be used to:
- ✗ Judge or stigmatize families
- ✗ Make discriminatory decisions
- ✗ Enforce language assimilation
- ✗ Pressure families about language choices

## Technical Specifications

**Dependencies:**
- Python 3.7+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

**Model Pipeline:**
1. Missing value imputation (median)
2. Feature scaling (StandardScaler)
3. Balanced class weights
4. 5-fold cross-validation
5. Test set evaluation
6. Model serialization (pickle)

**File Formats:**
- Input: CSV with headers
- Output: CSV with predictions + probabilities
- Model: Python pickle (.pkl)
- Visualizations: PNG (300 DPI)

## Next Steps for Users

1. **Pilot Testing**: Collect 50-100 samples to test the system
2. **Feature Validation**: Verify which features matter for YOUR population
3. **Full Deployment**: Scale to 300+ samples for robust predictions
4. **Ongoing Updates**: Retrain quarterly with new data
5. **Results Dissemination**: Share findings with community stakeholders

## Credits

**Theoretical Foundation:**
- Heritage language acquisition research
- Family language policy theory
- Community language vitality frameworks
- Second language acquisition studies

**Technical Implementation:**
- scikit-learn machine learning library
- pandas data manipulation
- matplotlib/seaborn visualization

## Version History

- **v1.0** (2025-11-12): Initial release
  - 20-feature model
  - 3 ML algorithms
  - 6 visualization types
  - Comprehensive documentation

## Contact & Support

For questions, issues, or contributions:
1. Review README.md and USAGE_GUIDE.md
2. Check data format against data_template.csv
3. Verify all dependencies installed correctly
4. Ensure sufficient training data (200+ samples minimum)

---

**Built for**: Research in language retention and heritage language maintenance
**Purpose**: Supporting bilingual development in immigrant communities
**License**: Educational and research use
**Status**: Production-ready, tested on synthetic data
