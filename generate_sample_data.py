#!/usr/bin/env python3
"""
Generate realistic synthetic data for language retention prediction.

This script creates a dataset with realistic correlations between features,
simulating real-world patterns observed in sociolinguistic research.
"""

import numpy as np
import pandas as pd
import os

def generate_language_retention_data(n_samples=500, random_state=42):
    """
    Generate synthetic language retention data with realistic correlations.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Generated dataset
    """
    np.random.seed(random_state)

    data = {}

    # Basic demographics
    data['child_id'] = range(1, n_samples + 1)
    data['age'] = np.random.randint(5, 19, n_samples)  # Ages 5-18
    data['birth_order'] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.35, 0.2, 0.05])

    # Parent English proficiency (correlated with years in country)
    data['parents_years_in_country'] = np.random.exponential(10, n_samples) + 5
    data['parents_years_in_country'] = np.clip(data['parents_years_in_country'], 5, 40)

    # English proficiency increases with years in country (with noise)
    base_proficiency = 1 + (data['parents_years_in_country'] / 10)
    data['parent1_english_proficiency'] = np.clip(
        base_proficiency + np.random.normal(0, 0.5, n_samples), 1, 5
    ).astype(int)
    data['parent2_english_proficiency'] = np.clip(
        base_proficiency + np.random.normal(0, 0.5, n_samples), 1, 5
    ).astype(int)

    # Family structure
    data['num_siblings'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.15, 0.35, 0.3, 0.15, 0.05])
    data['grandparents_in_household'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    # Education (higher education slightly correlated with higher English proficiency)
    data['parent1_education'] = np.clip(
        data['parent1_english_proficiency'] + np.random.randint(-1, 2, n_samples), 1, 5
    )
    data['parent2_education'] = np.clip(
        data['parent2_english_proficiency'] + np.random.randint(-1, 2, n_samples), 1, 5
    )

    # Community factors
    # Distance to ethnic enclave (exponential distribution - most people near, some far)
    data['distance_to_ethnic_enclave_km'] = np.random.exponential(8, n_samples)
    data['distance_to_ethnic_enclave_km'] = np.clip(data['distance_to_ethnic_enclave_km'], 0.1, 50)

    # Percentage same ethnicity (inversely correlated with distance to enclave)
    base_pct = 60 - (data['distance_to_ethnic_enclave_km'] * 2)
    data['pct_same_ethnicity_neighborhood'] = np.clip(
        base_pct + np.random.normal(0, 10, n_samples), 0, 100
    )

    # Heritage language school more available near enclaves
    prob_school = 1 / (1 + np.exp((data['distance_to_ethnic_enclave_km'] - 10) / 3))
    data['heritage_language_school_available'] = np.random.binomial(1, prob_school)

    # Years of heritage language instruction (depends on school availability and age)
    data['years_heritage_language_instruction'] = np.where(
        data['heritage_language_school_available'] == 1,
        np.random.randint(0, np.minimum(data['age'] - 4, 10), n_samples),
        0
    )

    # Religious institution attendance (more common in ethnic enclaves)
    prob_religious = 0.4 + (data['pct_same_ethnicity_neighborhood'] / 200)
    data['attends_religious_institution'] = np.random.binomial(1, prob_religious)

    # Heritage language services (correlated with attending religious institution)
    data['religious_services_in_heritage_lang'] = np.where(
        data['attends_religious_institution'] == 1,
        np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        0
    )

    # Bilingual education (less common overall)
    data['bilingual_education_program'] = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])

    # Social factors
    # Friends from same ethnicity correlated with neighborhood composition
    data['pct_friends_same_ethnicity'] = np.clip(
        data['pct_same_ethnicity_neighborhood'] + np.random.normal(0, 15, n_samples), 0, 100
    )

    # Heritage media consumption
    data['hours_per_week_heritage_media'] = np.random.exponential(3, n_samples)
    data['hours_per_week_heritage_media'] = np.clip(data['hours_per_week_heritage_media'], 0, 20)

    # Visits to heritage country
    data['visits_to_heritage_country'] = np.random.poisson(2, n_samples)
    data['visits_to_heritage_country'] = np.clip(data['visits_to_heritage_country'], 0, 15)

    # Parents speak heritage at home (inversely correlated with English proficiency)
    avg_eng_prof = (data['parent1_english_proficiency'] + data['parent2_english_proficiency']) / 2
    prob_heritage_home = 1 - (avg_eng_prof - 1) / 4  # Scale 0-1
    prob_heritage_home = np.clip(prob_heritage_home, 0.1, 0.9)
    data['parents_speak_heritage_at_home'] = np.random.binomial(1, prob_heritage_home)

    # TARGET VARIABLE: Language retention
    # Complex interaction of factors

    # Base probability from family language use
    retention_prob = data['parents_speak_heritage_at_home'] * 0.3

    # Boost from low parent English proficiency
    retention_prob += (5 - avg_eng_prof) / 10

    # Boost from grandparents
    retention_prob += data['grandparents_in_household'] * 0.15

    # Boost from ethnic enclave proximity
    retention_prob += (1 / (1 + data['distance_to_ethnic_enclave_km'] / 5)) * 0.15

    # Boost from neighborhood composition
    retention_prob += (data['pct_same_ethnicity_neighborhood'] / 100) * 0.15

    # Boost from formal instruction
    retention_prob += (data['years_heritage_language_instruction'] / 10) * 0.2

    # Boost from religious institution
    retention_prob += data['religious_services_in_heritage_lang'] * 0.1

    # Boost from social network
    retention_prob += (data['pct_friends_same_ethnicity'] / 100) * 0.1

    # Boost from media consumption
    retention_prob += np.minimum(data['hours_per_week_heritage_media'] / 20, 0.1)

    # Boost from heritage country visits
    retention_prob += np.minimum(data['visits_to_heritage_country'] / 15, 0.1)

    # Boost from siblings (more practice)
    retention_prob += np.minimum(data['num_siblings'] / 10, 0.1)

    # Age effect (older children might have had more exposure, but also more assimilation pressure)
    # Slight inverse U-shape
    age_factor = -0.01 * (data['age'] - 12) ** 2 / 10
    retention_prob += age_factor

    # Clip to valid probability range
    retention_prob = np.clip(retention_prob, 0.05, 0.95)

    # Generate binary outcome
    data['speaks_parent_language'] = np.random.binomial(1, retention_prob)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Round numeric columns appropriately
    df['parents_years_in_country'] = df['parents_years_in_country'].round(1)
    df['distance_to_ethnic_enclave_km'] = df['distance_to_ethnic_enclave_km'].round(1)
    df['pct_same_ethnicity_neighborhood'] = df['pct_same_ethnicity_neighborhood'].round(0).astype(int)
    df['pct_friends_same_ethnicity'] = df['pct_friends_same_ethnicity'].round(0).astype(int)
    df['hours_per_week_heritage_media'] = df['hours_per_week_heritage_media'].round(1)

    return df

def main():
    """Generate and save sample data."""
    print("Generating synthetic language retention data...")

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Generate data
    df = generate_language_retention_data(n_samples=500, random_state=42)

    # Save to CSV
    output_path = 'data/language_retention_data.csv'
    df.to_csv(output_path, index=False)

    print(f"\n✓ Generated {len(df)} samples")
    print(f"✓ Saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"\nTarget variable distribution:")
    print(f"  Speaks parent language: {df['speaks_parent_language'].sum()} ({df['speaks_parent_language'].mean()*100:.1f}%)")
    print(f"  Does not speak parent language: {(1-df['speaks_parent_language']).sum()} ({(1-df['speaks_parent_language'].mean())*100:.1f}%)")

    print(f"\nAge range: {df['age'].min()} - {df['age'].max()} years")
    print(f"Average parents' years in country: {df['parents_years_in_country'].mean():.1f} years")
    print(f"Average distance to ethnic enclave: {df['distance_to_ethnic_enclave_km'].mean():.1f} km")
    print(f"Families speaking heritage language at home: {df['parents_speak_heritage_at_home'].sum()} ({df['parents_speak_heritage_at_home'].mean()*100:.1f}%)")

    print("\n" + "="*60)
    print("\nFirst few rows:")
    print(df.head(3).to_string())

if __name__ == "__main__":
    main()
