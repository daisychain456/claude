# Language Retention Predictor - Usage Guide

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data (for testing/demonstration)

```bash
python generate_sample_data.py
```

This creates `data/language_retention_data.csv` with 500 synthetic samples.

### 3. Train the Model

```bash
python train_model.py --data data/language_retention_data.csv
```

Expected output:
- Model comparison table showing performance of all models
- Best model selection
- Feature importance ranking
- Saved model in `models/language_retention_model.pkl`
- Results in `results/model_comparison.csv`

### 4. Generate Visualizations

```bash
python visualize_results.py
```

This creates 6 visualization files in the `results/` directory:
1. **feature_importance.png** - Top 15 most predictive features
2. **confusion_matrix.png** - Model prediction accuracy breakdown
3. **roc_curve.png** - Model discrimination ability
4. **demographic_patterns.png** - Feature distributions by retention status
5. **correlation_heatmap.png** - Feature correlations
6. **retention_by_categories.png** - Retention rates by key factors

### 5. Make Predictions on New Data

```bash
python predict.py --input new_data.csv --output predictions.csv
```

The output CSV will include:
- All original columns
- `predicted_speaks_parent_language` (0 or 1)
- `probability_no` (probability of not speaking)
- `probability_yes` (probability of speaking)
- `confidence` (model confidence in prediction)
- `interpretation` (human-readable prediction)

---

## Working with Your Own Data

### Data Collection

1. Use `data_template.csv` as your starting template
2. Each row represents one child
3. Fill in all columns with accurate information

### Required Columns

Your CSV must include these columns (see README.md for detailed descriptions):

**Demographics:**
- child_id
- age
- birth_order

**Family Environment:**
- parent1_english_proficiency
- parent2_english_proficiency
- parents_years_in_country
- num_siblings
- grandparents_in_household
- parent1_education
- parent2_education
- parents_speak_heritage_at_home

**Community:**
- distance_to_ethnic_enclave_km
- pct_same_ethnicity_neighborhood
- heritage_language_school_available

**Education:**
- years_heritage_language_instruction
- attends_religious_institution
- religious_services_in_heritage_lang
- bilingual_education_program

**Social:**
- pct_friends_same_ethnicity
- hours_per_week_heritage_media
- visits_to_heritage_country

**Target (for training only):**
- speaks_parent_language

### Data Preparation Tips

1. **Missing Values**: The model handles missing values by filling them with median values from training data. However, minimize missing data for best results.

2. **Encoding**:
   - Binary variables: Use 0/1 (e.g., 0 = No, 1 = Yes)
   - Proficiency scales: Use 1-5 (1 = lowest, 5 = highest)
   - Education levels: 1 = elementary, 2 = middle school, 3 = high school, 4 = bachelor's, 5 = graduate degree
   - Percentages: Use 0-100 range

3. **Units**:
   - Distance: kilometers
   - Time: years (can be decimal, e.g., 2.5 years)
   - Age: whole years

---

## Advanced Usage

### Retraining with New Data

When you collect more data, retrain the model:

```bash
# Combine old and new data
python train_model.py --data combined_data.csv --output models/updated_model.pkl
```

### Custom Model Parameters

Modify model hyperparameters in `train_model.py`:

```python
# Example: Change Random Forest parameters
'Random Forest': RandomForestClassifier(
    n_estimators=200,  # More trees
    max_depth=15,      # Deeper trees
    random_state=42
)
```

### Using Different Test/Train Splits

```bash
python train_model.py --data your_data.csv --test-size 0.3  # 30% test set
```

### Batch Predictions

For large prediction jobs:

```python
# In Python script or notebook
import pandas as pd
from predict import load_model, make_predictions

model_data = load_model('models/language_retention_model.pkl')
df = pd.read_csv('large_dataset.csv')
results = make_predictions(df, model_data)
results.to_csv('batch_predictions.csv', index=False)
```

---

## Interpreting Results

### Feature Importance

Higher values indicate stronger predictive power. The top features typically include:
- Parents speaking heritage language at home
- Parent English proficiency
- Distance to ethnic enclave
- Heritage language instruction years

### Model Performance Metrics

- **Accuracy**: Overall correctness (aim for >75%)
- **Precision**: Of predicted "speaks", how many actually speak? (aim for >70%)
- **Recall**: Of children who speak, how many did we identify? (aim for >70%)
- **ROC-AUC**: Discrimination ability (aim for >0.80)

### Prediction Confidence

- **High confidence (>80%)**: Very reliable prediction
- **Medium confidence (60-80%)**: Moderately reliable
- **Low confidence (<60%)**: Uncertain, consider gathering more information

---

## Example Workflow

### Scenario: Research Study

1. **Design data collection**
   ```bash
   # Review template
   cat data_template.csv
   ```

2. **Collect pilot data** (n=50-100)
   - Survey families
   - Record in CSV following template

3. **Initial model training**
   ```bash
   python train_model.py --data pilot_data.csv
   python visualize_results.py
   ```

4. **Review feature importance**
   - Identify most predictive factors
   - Consider focusing data collection on top features

5. **Collect full dataset** (n=300-500+)

6. **Final model training**
   ```bash
   python train_model.py --data full_data.csv
   python visualize_results.py
   ```

7. **Make predictions for new cases**
   ```bash
   python predict.py --input new_families.csv
   ```

8. **Analyze patterns**
   - Review visualizations
   - Identify community-level trends
   - Inform policy recommendations

---

## Troubleshooting

### "Missing required columns" error

**Cause**: Your data CSV is missing required feature columns

**Solution**: Compare your columns with `data_template.csv`. Add missing columns.

### Poor model performance (accuracy <60%)

**Possible causes**:
- Insufficient data (need 200+ samples minimum)
- Data quality issues (inaccurate responses)
- Unbalanced classes (e.g., 90% speak, 10% don't)

**Solutions**:
- Collect more data
- Verify data accuracy
- For class imbalance, the model uses `class_weight='balanced'` automatically

### Low confidence predictions

**Cause**: Borderline cases or contradictory feature patterns

**Example**: Child has high heritage language exposure at home but lives far from ethnic enclave with low same-ethnicity neighborhood composition

**Solution**: Gather additional qualitative data for borderline cases

### Features have unexpected importance

**Cause**: Your population may have different patterns than the general model

**Solution**: This is normal! Different communities have different dynamics. Use feature importance to understand YOUR specific population.

---

## Best Practices

1. **Data Quality First**: Accurate data is more important than quantity
2. **Start Small**: Test with pilot data before full collection
3. **Iterate**: Retrain as you collect more data
4. **Context Matters**: Interpret results within your specific cultural/community context
5. **Ethical Use**: Use predictions to support families, not to judge them
6. **Documentation**: Keep notes on data collection methods and any special circumstances

---

## Support and Citation

### Questions or Issues?

1. Review this guide and README.md
2. Check that your data follows the template format
3. Verify all dependencies are installed correctly

### Using this in Research?

Please consider these citation elements:
- Acknowledge the sociolinguistic research basis
- Document your specific population and context
- Note any modifications to features or methodology
- Share insights about which features mattered most in your community

---

## Extending the Model

### Adding New Features

1. Add column to your data CSV
2. Update `feature_names` in preprocessing if needed
3. Retrain model - it will automatically include new features

### Example: Adding "Years in Heritage Language Daycare"

```csv
child_id,...,years_heritage_daycare,speaks_parent_language
1,...,2,1
2,...,0,0
```

The model will automatically incorporate this feature!

### Trying Different Models

Edit `train_model.py` to add new algorithms:

```python
from sklearn.svm import SVC

model_configs = {
    # ... existing models ...
    'SVM': SVC(probability=True, random_state=42)
}
```

---

## Performance Expectations

With synthetic data (500 samples):
- **Accuracy**: 75-85%
- **ROC-AUC**: 0.82-0.88
- **Training time**: 5-10 seconds

With real-world data, expect:
- **Lower accuracy initially** (60-70%) - this is normal!
- **Improvement with more data** (aim for 300+ samples)
- **Different feature importance** based on your community
