import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
# import shap  # Commented out to avoid dependency issues
import matplotlib.ticker as mtick

# Set the style for visualizations
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Create output directory for visualizations
os.makedirs('data/healthcare/visualizations', exist_ok=True)

def analyze_diabetes_data():
    """
    Perform comprehensive analysis on the Pima Indians Diabetes Dataset.
    
    This function loads the diabetes dataset, performs exploratory data analysis,
    visualizes key patterns, handles missing values, builds and evaluates machine
    learning models, and analyzes feature importance using both traditional and
    advanced explainability techniques.
    
    The function saves the processed data and visualizations to specified directories.
    
    Returns:
        None
    """
    print("\n" + "="*50)
    print("ANALYZING DIABETES DATASET")
    print("="*50 + "\n")
    
    # Load the dataset
    try:
        df = pd.read_csv('data/healthcare/datasets/diabetes.csv')
        print(f"Successfully loaded diabetes dataset")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
        print("\nData information:")
        print(df.info())
        print("\nBasic statistics:")
        print(df.describe())
        
        # Check for missing values
        print("\nMissing values per column:")
        print(df.isnull().sum())
        
        # Handle potential zero values in specific columns that should be non-zero
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in zero_columns:
            # Count zeros before replacement
            zeros_count = (df[column] == 0).sum()
            percent_zeros = (zeros_count / len(df)) * 100
            print(f"{column}: {zeros_count} zeros ({percent_zeros:.2f}%)")
            
            # Replace zeros with NaN for later imputation
            df[column] = df[column].replace(0, np.nan)
        
        # Enhanced Visualization - Distribution of Target Variable
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='Outcome', data=df, palette=['#5AB4DC', '#d65f5f'])
        
        # Add count and percentage annotations
        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            percentage = 100 * height / total
            ax.annotate(f'{int(height)}\n({percentage:.1f}%)',
                       (p.get_x() + p.get_width() / 2., height / 2),
                       ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        
        plt.title('Distribution of Diabetes Outcome', fontsize=16, fontweight='bold')
        plt.xlabel('Outcome (0: No Diabetes, 1: Diabetes)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks([0, 1], ['Non-Diabetic', 'Diabetic'], fontsize=10)
        plt.savefig('data/healthcare/visualizations/diabetes_outcome_distribution.png', dpi=300, bbox_inches='tight')
        
        # Distribution of numerical features with statistics
        plt.figure(figsize=(20, 15))
        for i, column in enumerate(df.columns[:-1], 1):
            plt.subplot(3, 3, i)
            sns.histplot(df[column].dropna(), kde=True, color='#5AB4DC')
            
            # Add mean and median lines
            mean_val = df[column].mean()
            median_val = df[column].median()
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
                        label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='green', linestyle='-', linewidth=1.5, 
                        label=f'Median: {median_val:.2f}')
            
            plt.title(f'Distribution of {column}', fontsize=12, fontweight='bold')
            plt.legend(fontsize=8)
            
        plt.tight_layout()
        plt.savefig('data/healthcare/visualizations/diabetes_feature_distributions.png', dpi=300, bbox_inches='tight')
        
        # Enhanced correlation matrix with annotations
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap=cmap, fmt='.2f',
                   linewidths=0.5, annot_kws={"size": 8}, vmin=-1, vmax=1)
        
        plt.title('Correlation Matrix of Diabetes Features', fontsize=16, fontweight='bold')
        plt.savefig('data/healthcare/visualizations/diabetes_correlation_matrix.png', dpi=300, bbox_inches='tight')
        
        # Prepare data for modeling
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Split data for modeling
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42, stratify=y)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Advanced modeling with ensemble approach
        # 1. Random Forest with hyperparameter tuning
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42)
        rf_cv = GridSearchCV(rf, rf_params, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
        rf_cv.fit(X_train_scaled, y_train)
        
        # 2. Gradient Boosting
        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(X_train_scaled, y_train)
        
        # 3. Ensemble (Voting Classifier)
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_cv.best_estimator_),
                ('gb', gb)
            ],
            voting='soft'
        )
        
        ensemble.fit(X_train_scaled, y_train)
        
        # Model evaluation
        y_pred = ensemble.predict(X_test_scaled)
        y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\nAdvanced Model Evaluation:")
        print(f"Best Random Forest Parameters: {rf_cv.best_params_}")
        print(f"Ensemble Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # ROC Curve and AUC
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='#5AB4DC', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig('data/healthcare/visualizations/diabetes_roc_curve.png', dpi=300, bbox_inches='tight')
        
        # Precision-Recall Curve
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, color='#d65f5f', lw=2, 
                 label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.axhline(y=sum(y_test)/len(y_test), color='gray', linestyle='--', 
                   label=f'Baseline (prevalence = {sum(y_test)/len(y_test):.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig('data/healthcare/visualizations/diabetes_precision_recall_curve.png', dpi=300, bbox_inches='tight')
        
        # Enhanced Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Create annotation text
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = [f"{value:,}" for value in cm.flatten()]
        group_percentages = [f"{value:.2%}" for value in cm.flatten()/np.sum(cm)]
        
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        
        # Plot heatmap with custom color scaling
        sns.heatmap(cm, annot=labels, fmt="", cmap='Blues', cbar=False,
                   annot_kws={"size": 12})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.xticks([0.5, 1.5], ['Non-Diabetic', 'Diabetic'], fontsize=10)
        plt.yticks([0.5, 1.5], ['Non-Diabetic', 'Diabetic'], fontsize=10, rotation=0)
        
        # Add metrics as text
        metrics_text = (
            f"Accuracy: {accuracy:.4f}\n"
            f"Sensitivity: {sensitivity:.4f}\n"
            f"Specificity: {specificity:.4f}\n"
            f"PPV: {ppv:.4f}\n"
            f"NPV: {npv:.4f}"
        )
        plt.figtext(0.15, 0.01, metrics_text, fontsize=12, ha="left")
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig('data/healthcare/visualizations/diabetes_confusion_matrix.png', dpi=300, bbox_inches='tight')
        
        # Feature Importance using Random Forest
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_cv.best_estimator_.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        
        # Add value labels
        for i, v in enumerate(feature_importance['Importance']):
            ax.text(v + 0.01, i, f"{v:.4f}", va='center', fontweight='bold')
            
        plt.title('Feature Importance for Diabetes Prediction', fontsize=16, fontweight='bold')
        plt.xlabel('Relative Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.xlim(0, feature_importance['Importance'].max() * 1.2)
        plt.tight_layout()
        plt.savefig('data/healthcare/visualizations/diabetes_feature_importance.png', dpi=300, bbox_inches='tight')
        
        # Advanced Visualization - Feature relationships with outcome
        top_features = feature_importance['Feature'].head(4).tolist()
        
        # Pairwise relationships
        plt.figure(figsize=(16, 14))
        
        for i, feature1 in enumerate(top_features):
            for j, feature2 in enumerate(top_features):
                plt.subplot(4, 4, i*4 + j + 1)
                
                if i == j:  # Histogram on diagonal
                    sns.histplot(data=df, x=feature1, hue='Outcome', element='step', 
                                common_norm=False, palette=['#5AB4DC', '#d65f5f'])
                    plt.title(f"{feature1}", fontsize=10)
                else:  # Scatter plot with regression line on non-diagonal
                    sns.scatterplot(data=df, x=feature1, y=feature2, hue='Outcome', 
                                   palette=['#5AB4DC', '#d65f5f'], alpha=0.7)
                    plt.title(f"{feature1} vs {feature2}", fontsize=10)
        
        plt.tight_layout()
        plt.savefig('data/healthcare/visualizations/diabetes_feature_relationships.png', dpi=300, bbox_inches='tight')
        
        # Save processed data
        df_processed = pd.DataFrame(X_imputed)
        df_processed['Outcome'] = y
        df_processed.to_csv('data/healthcare/datasets/diabetes_processed.csv', index=False)
        
        print("\nProcessed diabetes data saved to data/healthcare/datasets/diabetes_processed.csv")
        print("Visualizations saved to data/healthcare/visualizations/")
        
    except Exception as e:
        print(f"Error analyzing diabetes dataset: {e}")
        import traceback
        traceback.print_exc()

def analyze_heart_disease_data():
    """
    Perform comprehensive analysis on the Cleveland Heart Disease Dataset.
    
    This function loads the heart disease dataset, performs exploratory data analysis,
    visualizes key patterns, handles missing values, builds and evaluates machine
    learning models, and analyzes feature importance.
    
    The function saves the processed data and visualizations to specified directories.
    
    Returns:
        None
    """
    print("\n" + "="*50)
    print("ANALYZING HEART DISEASE DATASET")
    print("="*50 + "\n")
    
    # Load the dataset
    try:
        # Cleveland heart disease dataset has no header
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = pd.read_csv('data/healthcare/datasets/heart_cleveland.csv', header=None, names=column_names)
        
        # Replace '?' with NaN
        df = df.replace('?', np.nan)
        
        # Convert columns to appropriate types
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
        
        print(f"Successfully loaded heart disease dataset")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:")
        print(df.head())
        print("\nData information:")
        print(df.info())
        print("\nBasic statistics:")
        print(df.describe())
        
        # Check for missing values
        print("\nMissing values per column:")
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        missing_info = pd.DataFrame({
            'Count': missing,
            'Percent': missing_percent
        })
        print(missing_info[missing_info['Count'] > 0])
        
        # Drop rows with missing values for simplicity
        df_clean = df.dropna()
        print(f"\nShape after dropping missing values: {df_clean.shape}")
        
        # Mapping dictionaries for interpretable values
        sex_map = {0: 'Female', 1: 'Male'}
        cp_map = {1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-anginal Pain', 4: 'Asymptomatic'}
        fbs_map = {0: 'False', 1: 'True'}
        restecg_map = {0: 'Normal', 1: 'ST-T Abnormality', 2: 'Left Ventricular Hypertrophy'}
        exang_map = {0: 'No', 1: 'Yes'}
        slope_map = {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'}
        thal_map = {3: 'Normal', 6: 'Fixed Defect', 7: 'Reversible Defect'}
        target_map = {0: 'No Disease', 1: 'Disease', 2: 'Disease', 3: 'Disease', 4: 'Disease'}
        
        # Enhanced visualizations
        # 1. Heart Disease Severity Distribution
        plt.figure(figsize=(12, 8))
        ax = sns.countplot(x='target', data=df_clean, palette='viridis')
        
        # Add count and percentage annotations
        total = len(df_clean)
        for p in ax.patches:
            height = p.get_height()
            percentage = 100 * height / total
            ax.annotate(f'{int(height)}\n({percentage:.1f}%)',
                       (p.get_x() + p.get_width() / 2., height / 2),
                       ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        
        plt.title('Distribution of Heart Disease Severity', fontsize=16, fontweight='bold')
        plt.xlabel('Heart Disease Severity (0: No disease, 1-4: Increasing severity)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.savefig('data/healthcare/visualizations/heart_disease_distribution.png', dpi=300, bbox_inches='tight')
        
        # Convert target to binary for simplicity (0: No disease, 1: Disease)
        df_clean['target_binary'] = df_clean['target'].apply(lambda x: 0 if x == 0 else 1)
        
        # Demographic Analysis
        # 2. Age distribution by heart disease with KDE
        plt.figure(figsize=(12, 8))
        
        sns.kdeplot(data=df_clean[df_clean['target_binary']==0], x='age', fill=True, 
                   color='#5AB4DC', alpha=0.7, label='No Disease')
        sns.kdeplot(data=df_clean[df_clean['target_binary']==1], x='age', fill=True,
                   color='#d65f5f', alpha=0.7, label='Disease')
        
        plt.title('Age Distribution by Heart Disease Status', fontsize=16, fontweight='bold')
        plt.xlabel('Age (years)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(title='Heart Disease Status', fontsize=12)
        
        # Add median age lines
        median_age_healthy = df_clean[df_clean['target_binary']==0]['age'].median()
        median_age_disease = df_clean[df_clean['target_binary']==1]['age'].median()
        
        plt.axvline(median_age_healthy, color='blue', linestyle='--', 
                   label=f'Median Age (Healthy): {median_age_healthy}')
        plt.axvline(median_age_disease, color='red', linestyle='--', 
                   label=f'Median Age (Disease): {median_age_disease}')
        
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig('data/healthcare/visualizations/heart_disease_age_distribution.png', dpi=300, bbox_inches='tight')
        
        # 3. Gender distribution with percentages
        plt.figure(figsize=(14, 8))
        
        # Create a cross-tabulation
        gender_disease = pd.crosstab(df_clean['sex'], df_clean['target_binary'], 
                                    normalize='index') * 100
        
        ax = gender_disease.plot(kind='bar', stacked=True, color=['#5AB4DC', '#d65f5f'], 
                               figsize=(12, 8))
        
        # Add count annotations
        gender_counts = df_clean['sex'].value_counts()
        gender_totals = pd.crosstab(df_clean['sex'], df_clean['target_binary'])
        
        # Add percentage annotations
        for i, (idx, row) in enumerate(gender_disease.iterrows()):
            # Count of no disease
            count_no = gender_totals.iloc[i, 0]
            pct_no = row[0]
            ax.annotate(f'{count_no}\n({pct_no:.1f}%)', 
                       (i-0.1, pct_no/2), 
                       ha='center', color='white', fontweight='bold')
            
            # Count of disease
            count_yes = gender_totals.iloc[i, 1]
            pct_yes = row[1]
            ax.annotate(f'{count_yes}\n({pct_yes:.1f}%)', 
                       (i-0.1, pct_no + pct_yes/2), 
                       ha='center', color='white', fontweight='bold')
        
        plt.title('Heart Disease by Gender', fontsize=16, fontweight='bold')
        plt.xlabel('Gender', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.xticks([0, 1], [sex_map[0], sex_map[1]], fontsize=12)
        plt.legend(['No Disease', 'Disease'], fontsize=12)
        
        # Add total counts
        for i, gender in enumerate([0, 1]):
            plt.annotate(f'Total: {gender_counts[gender]}', 
                        (i-0.1, -5), 
                        ha='center', fontsize=11)
            
        plt.ylim(0, 110)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('data/healthcare/visualizations/heart_disease_by_gender.png', dpi=300, bbox_inches='tight')
        
        # 4. Enhanced correlation matrix
        plt.figure(figsize=(16, 14))
        numeric_df = df_clean.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap=cmap, fmt='.2f',
                   linewidths=0.5, annot_kws={"size": 10}, vmin=-1, vmax=1)
        
        plt.title('Correlation Matrix of Heart Disease Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data/healthcare/visualizations/heart_disease_correlation_matrix.png', dpi=300, bbox_inches='tight')
        
        # 5. Chest Pain Type Distribution by Heart Disease
        plt.figure(figsize=(14, 8))
        cp_counts = pd.crosstab(df_clean['cp'], df_clean['target_binary'], normalize='index') * 100
        
        ax = cp_counts.plot(kind='bar', stacked=True, color=['#5AB4DC', '#d65f5f'], figsize=(12, 8))
        
        plt.title('Chest Pain Type by Heart Disease Status', fontsize=16, fontweight='bold')
        plt.xlabel('Chest Pain Type', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.xticks(range(4), [cp_map[i] for i in sorted(cp_map.keys())], fontsize=10, rotation=30, ha='right')
        plt.legend(['No Disease', 'Disease'], fontsize=12, title='Heart Disease')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('data/healthcare/visualizations/heart_disease_by_chest_pain.png', dpi=300, bbox_inches='tight')
        
        # 6. Advanced modeling
        # Prepare data for modeling
        X = df_clean.drop(['target', 'target_binary'], axis=1)
        y = df_clean['target_binary']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Define parameter grid
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
        
        # Perform grid search
        grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate model
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        print("\nAdvanced Model Evaluation:")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # ROC Curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='#5AB4DC', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig('data/healthcare/visualizations/heart_disease_roc_curve.png', dpi=300, bbox_inches='tight')
        
        # Feature importance
        feature_importances = best_model.named_steps['classifier'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
        
        # Add value labels
        for i, v in enumerate(feature_importance_df['Importance']):
            ax.text(v + 0.01, i, f"{v:.4f}", va='center', fontweight='bold')
        
        plt.title('Feature Importance for Heart Disease Prediction', fontsize=16, fontweight='bold')
        plt.xlabel('Relative Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.xlim(0, feature_importance_df['Importance'].max() * 1.2)
        plt.tight_layout()
        plt.savefig('data/healthcare/visualizations/heart_disease_feature_importance.png', dpi=300, bbox_inches='tight')
        
        # Enhanced Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Create annotation text
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = [f"{value:,}" for value in cm.flatten()]
        group_percentages = [f"{value:.2%}" for value in cm.flatten()/np.sum(cm)]
        
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        
        # Plot heatmap with custom color scaling
        sns.heatmap(cm, annot=labels, fmt="", cmap='Blues', cbar=False,
                   annot_kws={"size": 12})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.xticks([0.5, 1.5], ['No Disease', 'Disease'], fontsize=10)
        plt.yticks([0.5, 1.5], ['No Disease', 'Disease'], fontsize=10, rotation=0)
        
        # Add metrics as text
        metrics_text = (
            f"Accuracy: {accuracy:.4f}\n"
            f"Sensitivity: {sensitivity:.4f}\n"
            f"Specificity: {specificity:.4f}\n"
            f"PPV: {ppv:.4f}\n"
            f"NPV: {npv:.4f}"
        )
        plt.figtext(0.15, 0.01, metrics_text, fontsize=12, ha="left")
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig('data/healthcare/visualizations/heart_disease_confusion_matrix.png', dpi=300, bbox_inches='tight')
        
        # Save processed data
        df_clean.to_csv('data/healthcare/datasets/heart_disease_processed.csv', index=False)
        print("\nProcessed heart disease data saved to data/healthcare/datasets/heart_disease_processed.csv")
        print("Visualizations saved to data/healthcare/visualizations/")
        
    except Exception as e:
        print(f"Error analyzing heart disease dataset: {e}")
        import traceback
        traceback.print_exc()

# Run the analysis
if __name__ == "__main__":
    analyze_diabetes_data()
    analyze_heart_disease_data() 