# Language Retention Predictor for Second-Generation Immigrant Children

A machine learning system to predict whether second-generation immigrant children will speak their parents' native/heritage language.

## Features Description

### Demographics
- **child_id**: Unique identifier for each child
- **age**: Child's age (years)
- **birth_order**: Birth order (1=first child, 2=second, etc.)

### Family Language Environment
- **parent1_english_proficiency**: Parent 1's English proficiency (1=minimal, 5=native-like)
- **parent2_english_proficiency**: Parent 2's English proficiency (1=minimal, 5=native-like)
- **parents_years_in_country**: Average years parents have lived in host country
- **num_siblings**: Number of siblings in household
- **grandparents_in_household**: Whether grandparents live in household (0=no, 1=yes)
- **parent1_education**: Parent 1's education level (1=elementary, 2=middle school, 3=high school, 4=bachelor's, 5=graduate degree)
- **parent2_education**: Parent 2's education level (same scale)
- **parents_speak_heritage_at_home**: Parents primarily speak heritage language at home (0=no, 1=yes)

### Community Factors
- **distance_to_ethnic_enclave_km**: Distance to nearest ethnic enclave in kilometers
- **pct_same_ethnicity_neighborhood**: Percentage of same ethnicity in neighborhood (0-100)
- **heritage_language_school_available**: Heritage language school available nearby (0=no, 1=yes)

### Educational & Institutional
- **years_heritage_language_instruction**: Years of formal heritage language instruction
- **attends_religious_institution**: Attends religious institution (0=no, 1=yes)
- **religious_services_in_heritage_lang**: Religious services conducted in heritage language (0=no, 1=yes)
- **bilingual_education_program**: Participated in bilingual education (0=no, 1=yes)

### Social & Cultural
- **pct_friends_same_ethnicity**: Percentage of friends from same ethnic background (0-100)
- **hours_per_week_heritage_media**: Hours per week consuming heritage language media
- **visits_to_heritage_country**: Number of visits to parents' country of origin

### Target Variable
- **speaks_parent_language**: Whether child speaks parent's language fluently (0=no, 1=yes)

## Files

- `data_template.csv`: Template for data collection with sample entries
- `generate_sample_data.py`: Generate realistic synthetic data for testing
- `train_model.py`: Train the language retention prediction model
- `predict.py`: Make predictions on new data
- `visualize_results.py`: Generate comprehensive visualizations
- `requirements.txt`: Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Sample Data (for testing)
```bash
python generate_sample_data.py
```

### 2. Train Model
```bash
python train_model.py --data data/language_retention_data.csv
```

### 3. Make Predictions
```bash
python predict.py --input new_data.csv --output predictions.csv
```

### 4. Generate Visualizations
```bash
python visualize_results.py
```

## Model Details

The system uses an ensemble approach:
- **Logistic Regression**: Baseline interpretable model
- **Random Forest**: Captures non-linear relationships
- **Gradient Boosting**: Best predictive performance

Features are automatically scaled and the model handles missing values appropriately.

## Assumptions

1. **Data Quality**: Survey responses are accurate and honest
2. **Language Proficiency Definition**: "Speaks parent language" means conversational fluency (can hold basic conversations)
3. **Age Range**: Model optimized for children ages 5-18
4. **Cultural Similarity**: Model may need adjustment for different cultural groups with different language retention patterns
5. **Temporal Stability**: Social/community factors are relatively stable over childhood

## Extending the Model

To add new data and retrain:

1. Add new rows to your data CSV following the template format
2. Run: `python train_model.py --data your_data.csv`
3. The model will automatically retrain and save updated version

To add new features:

1. Add column(s) to your data CSV
2. Update the feature lists in `train_model.py` if needed
3. Retrain the model

## Ethical Considerations

This model is designed for research and understanding language retention patterns. It should:
- NOT be used to judge families or communities
- NOT be used for discriminatory purposes
- Be used to understand and support heritage language maintenance
- Inform educational policy and community resource allocation

## References

This feature set is based on sociolinguistic research on heritage language retention, including:
- Family language policy research
- Second language acquisition theory
- Community language vitality frameworks
