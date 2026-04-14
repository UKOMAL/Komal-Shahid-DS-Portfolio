# Data Directory

This directory contains datasets for the treatment-seeking prediction project.

## Structure

### `raw/`
**Contains:** Original, unmodified dataset from OSMI survey
- `OSMI_2016_survey.csv` — Raw survey responses (1,259 rows × 25 columns)
- **Source**: Open Source Mental Illness (OSMI) 2016 Mental Health in Tech Survey
- **License**: Creative Commons Attribution-ShareAlike 4.0 International
- **Format**: CSV (comma-separated values)

**Fields in raw data:**
- Demographic: `age`, `gender`, `country`, `state`
- Work environment: `tech_company`, `company_size`, `benefits`, `care_options`
- Health: `family_history`, `diagnosed_condition`, `work_interfere`
- **Target**: `treatment` (binary: yes/no for seeking treatment)

### `processed/`
**Contains:** Cleaned, preprocessed dataset ready for modeling
- `treatment_seeking_clean.csv` — After imputation, encoding, and feature engineering
- `train_set.csv` — 80% training split (stratified)
- `test_set.csv` — 20% test split (stratified)
- `features_metadata.json` — Data dictionary describing all features

**Processing applied:**
1. Missing value handling (mode for categorical, median for numerical)
2. One-hot encoding of categorical variables
3. Standardization/scaling for logistic regression
4. SMOTE balancing applied to training set only
5. Train/test split with stratification to preserve class distribution

## Data Usage Instructions

### Loading Raw Data
```python
import pandas as pd
df = pd.read_csv('raw/OSMI_2016_survey.csv')
print(df.shape)  # (1259, 25)
print(df.info()) # Data types and missing values
```

### Loading Processed Data
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('processed/treatment_seeking_clean.csv')
X = df.drop('treatment', axis=1)
y = df['treatment']

# Or use pre-split files:
X_train = pd.read_csv('processed/train_set.csv').drop('treatment', axis=1)
y_train = pd.read_csv('processed/train_set.csv')['treatment']
```

## Data Ethics & Privacy

- **De-identified**: No personally identifiable information (names, emails, etc.)
- **Voluntary**: Respondents voluntarily completed the OSMI survey
- **Public domain**: OSMI survey data is publicly available under Creative Commons license
- **Aggregate analysis**: All reporting is at group level, no individual predictions

## Data Quality Notes

- **Class imbalance**: 63% did not seek treatment, 37% did (addressed with SMOTE)
- **Missing values**: ~5-10% across features (handled via imputation)
- **Outliers**: No extreme outliers detected; categorical responses bounded by survey design
- **Temporal**: 2016 survey data; organizational attitudes toward mental health may have evolved

## Citation

If using this dataset, please cite:
```
Open Source Mental Illness. (2016). Mental health in tech survey. 
Retrieved from https://www.osmihelp.org/
```

## Questions?

For questions about data quality or processing, see:
- Main project README: `../README.md`
- Analysis code: `../code/shahid_dsc680_milestone2_analysis.py`
- EDA notebook: `../notebooks/01_eda_exploratory_analysis.ipynb`
