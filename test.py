
# =============================================================================
# PARTIE 1: IMPORTATION DES LIBRAIRIES
# =============================================================================
print("=" * 80)
print("PARTIE 1: IMPORTATION DES LIBRAIRIES")
print("=" * 80)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import joblib
from datetime import datetime
warnings.filterwarnings('ignore')

# Librairies de Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_curve, auc, f1_score, mean_squared_error, r2_score,
                             silhouette_score, jaccard_score, mean_absolute_error)

# Modèles de Régression
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Modèles de Classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Modèles de Clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Toutes les librairies importées avec succès!\n")

# Création de la structure de dossiers
def create_project_structure():
    """Crée la structure de dossiers pour le projet"""
    folders = [
        'data',
        'models',
        'models/regression',
        'models/classification',
        'models/clustering',
        'models/dimensionality_reduction',
        'models/recommendation',
        'visualizations',
        'results',
        'notebooks'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print("✓ Structure de dossiers créée:")
    for folder in folders:
        print(f"  - {folder}/")
    print()

create_project_structure()

# =============================================================================
# PARTIE 2: CHARGEMENT ET EXPLORATION DES DONNÉES (EDA)
# =============================================================================
print("=" * 80)
print("PARTIE 2: CHARGEMENT ET EXPLORATION DES DONNÉES")
print("=" * 80)

# Chargement du dataset
try:
    df = pd.read_csv('data/Student Mental health.csv')
    print("✓ Dataset chargé depuis le fichier local")
except FileNotFoundError:
    print("⚠ Fichier non trouvé. Création d'un dataset d'exemple...")
    # Création d'un dataset d'exemple pour la démonstration
    np.random.seed(42)
    n_samples = 300
    
    df = pd.DataFrame({
        'Timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='H'),
        'Choose your gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 26, n_samples),
        'What is your course?': np.random.choice(['Engineering', 'Business', 'Science', 'Arts'], n_samples),
        'Your current year of Study': np.random.choice(['Year 1', 'Year 2', 'Year 3', 'Year 4'], n_samples),
        'What is your CGPA?': np.random.uniform(2.0, 4.0, n_samples),
        'Marital status': np.random.choice(['Single', 'Married'], n_samples, p=[0.9, 0.1]),
        'Do you have Depression?': np.random.choice(['Yes', 'No'], n_samples),
        'Do you have Anxiety?': np.random.choice(['Yes', 'No'], n_samples),
        'Do you have Panic attack?': np.random.choice(['Yes', 'No'], n_samples),
        'Did you seek any specialist for a treatment?': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    })
    df.to_csv('data/Student Mental health.csv', index=False)
    print("✓ Dataset d'exemple créé et sauvegardé")

print(f"\n✓ Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes\n")

# Affichage des premières lignes
print("--- Aperçu des données ---")
print(df.head(10))
print("\n")

# Informations sur le dataset
print("--- Informations sur le dataset ---")
print(df.info())
print("\n")

# Statistiques descriptives
print("--- Statistiques descriptives ---")
print(df.describe(include='all'))
print("\n")

# Vérification des valeurs manquantes
print("--- Valeurs manquantes ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing[missing > 0].any() else "Aucune valeur manquante")
print(f"\nTotal de valeurs manquantes: {df.isnull().sum().sum()}")

# Sauvegarde des statistiques
stats_summary = {
    'shape': df.shape,
    'columns': list(df.columns),
    'dtypes': df.dtypes.astype(str).to_dict(),
    'missing_values': df.isnull().sum().to_dict(),
    'numeric_stats': df.describe().to_dict()
}

with open('results/data_summary.txt', 'w', encoding='utf-8') as f:
    f.write("RÉSUMÉ DES DONNÉES - STUDENT MENTAL HEALTH\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Forme du dataset: {stats_summary['shape']}\n")
    f.write(f"Colonnes: {stats_summary['columns']}\n")
    f.write(f"Valeurs manquantes totales: {df.isnull().sum().sum()}\n")

print("✓ Résumé des données sauvegardé dans 'results/data_summary.txt'\n")

# =============================================================================
# PARTIE 3: VISUALISATION DES DONNÉES (EDA)
# =============================================================================
print("=" * 80)
print("PARTIE 3: VISUALISATION DES DONNÉES")
print("=" * 80)

# Nettoyage des noms de colonnes pour faciliter la manipulation
df.columns = df.columns.str.strip()

# Visualisations principales
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Distribution du genre
if 'Choose your gender' in df.columns:
    df['Choose your gender'].value_counts().plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Distribution du Genre', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Genre')
    axes[0, 0].set_ylabel('Nombre d\'étudiants')
    axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Distribution de l'âge
if 'Age' in df.columns:
    axes[0, 1].hist(df['Age'].dropna(), bins=15, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Distribution de l\'Âge', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Âge')
    axes[0, 1].set_ylabel('Fréquence')

# 3. Distribution du CGPA
if 'What is your CGPA?' in df.columns:
    axes[0, 2].hist(df['What is your CGPA?'].dropna(), bins=20, color='lightgreen', edgecolor='black')
    axes[0, 2].set_title('Distribution du CGPA', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('CGPA')
    axes[0, 2].set_ylabel('Fréquence')

# 4. Dépression
if 'Do you have Depression?' in df.columns:
    df['Do you have Depression?'].value_counts().plot(kind='pie', ax=axes[1, 0], autopct='%1.1f%%')
    axes[1, 0].set_title('Prévalence de la Dépression', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('')

# 5. Anxiété
if 'Do you have Anxiety?' in df.columns:
    df['Do you have Anxiety?'].value_counts().plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
    axes[1, 1].set_title('Prévalence de l\'Anxiété', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('')

# 6. Attaques de panique
if 'Do you have Panic attack?' in df.columns:
    df['Do you have Panic attack?'].value_counts().plot(kind='pie', ax=axes[1, 2], autopct='%1.1f%%', colors=['#99ff99', '#ffcc99'])
    axes[1, 2].set_title('Prévalence des Attaques de Panique', fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel('')

plt.tight_layout()
plt.savefig('visualizations/01_eda_overview.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualisations générées et sauvegardées!\n")

# Visualisation supplémentaire: Problèmes de santé mentale par année d'étude
if all(col in df.columns for col in ['Your current year of Study', 'Do you have Depression?']):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    mental_health_by_year = pd.crosstab(
        df['Your current year of Study'],
        df['Do you have Depression?'],
        normalize='index'
    ) * 100
    
    mental_health_by_year.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_title('Prévalence de la Dépression par Année d\'Étude', fontsize=16, fontweight='bold')
    ax.set_xlabel('Année d\'Étude')
    ax.set_ylabel('Pourcentage (%)')
    ax.legend(['Non', 'Oui'], title='Dépression')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/02_depression_by_year.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# PARTIE 4: NETTOYAGE ET PRÉPARATION DES DONNÉES
# =============================================================================
print("=" * 80)
print("PARTIE 4: NETTOYAGE ET PRÉPARATION DES DONNÉES")
print("=" * 80)

# 4.1 Copie du dataframe
df_clean = df.copy()

# 4.2 Gestion des valeurs manquantes
print("--- 4.1 Traitement des valeurs manquantes ---")
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].dtype in ['float64', 'int64']:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"✓ {col}: {df[col].isnull().sum()} valeurs imputées avec la médiane ({median_val:.2f})")
        else:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"✓ {col}: {df[col].isnull().sum()} valeurs imputées avec le mode ({mode_val})")

print(f"\nValeurs manquantes restantes: {df_clean.isnull().sum().sum()}")

# 4.3 Encodage des variables catégorielles
print("\n--- 4.2 Encodage des variables catégorielles ---")

label_encoders = {}
categorical_columns = df_clean.select_dtypes(include=['object']).columns

# Exclure la colonne Timestamp si elle existe
categorical_columns = [col for col in categorical_columns if 'Timestamp' not in col]

for col in categorical_columns:
    le = LabelEncoder()
    df_clean[col + '_encoded'] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le
    print(f"✓ {col}: {len(le.classes_)} catégories encodées")

# Sauvegarde des encoders
joblib.dump(label_encoders, 'models/label_encoders.pkl')
print("\n✓ Label encoders sauvegardés dans 'models/label_encoders.pkl'")

# 4.4 Création de variables cibles
print("\n--- 4.3 Création des variables cibles ---")

# Variable cible pour la classification (présence d'au moins un problème de santé mentale)
if all(col in df_clean.columns for col in ['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?']):
    df_clean['mental_health_issue'] = (
        (df_clean['Do you have Depression?'] == 'Yes') |
        (df_clean['Do you have Anxiety?'] == 'Yes') |
        (df_clean['Do you have Panic attack?'] == 'Yes')
    ).astype(int)
    print(f"✓ Variable 'mental_health_issue' créée (0: Aucun problème, 1: Au moins un problème)")
    print(f"  Distribution: {df_clean['mental_health_issue'].value_counts().to_dict()}")

# Score de santé mentale pour la régression (nombre de problèmes)
if all(col in df_clean.columns for col in ['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?']):
    df_clean['mental_health_score'] = (
        (df_clean['Do you have Depression?'] == 'Yes').astype(int) +
        (df_clean['Do you have Anxiety?'] == 'Yes').astype(int) +
        (df_clean['Do you have Panic attack?'] == 'Yes').astype(int)
    )
    print(f"✓ Variable 'mental_health_score' créée (0-3: nombre de problèmes)")
    print(f"  Distribution: {df_clean['mental_health_score'].value_counts().sort_index().to_dict()}")

# 4.5 Sélection des features pour le modèle
print("\n--- 4.4 Sélection des features ---")

# Features numériques et encodées
feature_columns = []
if 'Age' in df_clean.columns:
    feature_columns.append('Age')
if 'What is your CGPA?' in df_clean.columns:
    feature_columns.append('What is your CGPA?')

# Ajouter les colonnes encodées
encoded_cols = [col for col in df_clean.columns if '_encoded' in col]
feature_columns.extend(encoded_cols)

# Exclure les variables cibles et originales non encodées
exclude_cols = ['mental_health_issue', 'mental_health_score', 'Timestamp',
                'Do you have Depression?', 'Do you have Anxiety?', 
                'Do you have Panic attack?', 'Did you seek any specialist for a treatment?']

feature_columns = [col for col in feature_columns if col not in exclude_cols]

print(f"Features sélectionnées: {feature_columns}")

# Création des matrices X et y
X = df_clean[feature_columns]
y_regression = df_clean['mental_health_score'] if 'mental_health_score' in df_clean.columns else None
y_classification = df_clean['mental_health_issue'] if 'mental_health_issue' in df_clean.columns else None

print(f"\n✓ Features (X): {X.shape}")
if y_regression is not None:
    print(f"✓ Target régression (y): {y_regression.shape}")
if y_classification is not None:
    print(f"✓ Target classification (y): {y_classification.shape}")

# 4.6 Détection des outliers
print("\n--- 4.5 Détection des outliers (méthode IQR) ---")

numerical_cols = X.select_dtypes(include=[np.number]).columns

fig, axes = plt.subplots(1, len(numerical_cols), figsize=(5*len(numerical_cols), 4))
if len(numerical_cols) == 1:
    axes = [axes]

for idx, col in enumerate(numerical_cols):
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers détectés")
    
    axes[idx].boxplot(X[col].dropna())
    axes[idx].set_title(f'Boxplot - {col}')
    axes[idx].set_ylabel('Valeur')

plt.tight_layout()
plt.savefig('visualizations/03_outliers_detection.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.6 Analyse de corrélation
print("\n--- 4.6 Analyse de corrélation ---")

# CORRECTION: Filtrer uniquement les colonnes numériques pour la corrélation
numeric_columns_for_corr = []
for col in feature_columns + (['mental_health_score'] if 'mental_health_score' in df_clean.columns else []):
    if pd.api.types.is_numeric_dtype(df_clean[col]):
        numeric_columns_for_corr.append(col)
    else:
        print(f"⚠ Colonne exclue de la corrélation (non numérique): {col}")

# Vérifier qu'on a des colonnes numériques pour la corrélation
if numeric_columns_for_corr:
    correlation_matrix = df_clean[numeric_columns_for_corr].corr()
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matrice de Corrélation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/04_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Variables les plus corrélées avec la cible
    if 'mental_health_score' in correlation_matrix.columns:
        print("\nCorrélations avec le score de santé mentale:")
        target_corr = correlation_matrix['mental_health_score'].sort_values(ascending=False)
        print(target_corr)
else:
    print("⚠ Aucune colonne numérique disponible pour l'analyse de corrélation")
    correlation_matrix = pd.DataFrame()

# 4.7 Normalisation des données
print("\n--- 4.7 Normalisation des données ---")

# CORRECTION: Convertir les colonnes non numériques en numériques
X_numeric = X.copy()
import re
# Fonction pour convertir CGPA en numérique
def convert_cgpa_to_numeric(cgpa_value):
    if isinstance(cgpa_value, (int, float)):
        return float(cgpa_value)
    elif isinstance(cgpa_value, str):
        # Extraire le premier nombre (ex: "3.00 - 3.49" -> 3.00)
        match = re.search(r'(\d+\.\d+)', str(cgpa_value))
        if match:
            return float(match.group(1))
        else:
            return np.nan
    else:
        return np.nan

# Appliquer la conversion à la colonne CGPA si elle existe
if 'What is your CGPA?' in X_numeric.columns:
    X_numeric['What is your CGPA?'] = X_numeric['What is your CGPA?'].apply(convert_cgpa_to_numeric)
    print("✓ Colonne CGPA convertie en valeurs numériques")

# Remplacer les valeurs manquantes par la médiane
X_numeric = X_numeric.apply(pd.to_numeric, errors='coerce')
X_numeric = X_numeric.fillna(X_numeric.median())

# Maintenant normaliser
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)
X_scaled = pd.DataFrame(X_scaled, columns=X_numeric.columns)

# Sauvegarde du scaler
joblib.dump(scaler, 'models/standard_scaler.pkl')
print("✓ Données normalisées avec StandardScaler")
print("✓ Scaler sauvegardé dans 'models/standard_scaler.pkl'")
# Sauvegarde des données préparées
df_clean.to_csv('data/student_mental_health_cleaned.csv', index=False)
X_scaled.to_csv('data/X_scaled.csv', index=False)
if y_regression is not None:
    y_regression.to_csv('data/y_regression.csv', index=False)
if y_classification is not None:
    y_classification.to_csv('data/y_classification.csv', index=False)

print("\n✓ Données nettoyées sauvegardées dans 'data/'")
print("\n--- Partie 1 complétée: Préparation des données ---\n")

# =============================================================================
# PARTIE 5: RÉGRESSION LINÉAIRE SIMPLE ET MULTIPLE
# =============================================================================
print("=" * 80)
print("PARTIE 5: RÉGRESSION LINÉAIRE")
print("=" * 80)

if y_regression is not None:
    # Division des données
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X_scaled, y_regression, test_size=0.2, random_state=42
    )
    
    print(f"Données d'entraînement: {X_train.shape}")
    print(f"Données de test: {X_test.shape}\n")
    
    # 5.1 Régression Linéaire Simple
    print("--- 5.1 Régression Linéaire Simple ---")
    
    # Choisir la meilleure feature basée sur la corrélation
    if 'What is your CGPA?' in X.columns:
        simple_feature = 'What is your CGPA?'
    else:
        simple_feature = X.columns[0]
    
    lr_simple = LinearRegression()
    lr_simple.fit(X_train[[simple_feature]], y_train_reg)
    y_pred_simple = lr_simple.predict(X_test[[simple_feature]])
    
    # Métriques
    mse_simple = mean_squared_error(y_test_reg, y_pred_simple)
    rmse_simple = np.sqrt(mse_simple)
    mae_simple = mean_absolute_error(y_test_reg, y_pred_simple)
    r2_simple = r2_score(y_test_reg, y_pred_simple)
    
    print(f"Feature utilisée: {simple_feature}")
    print(f"Coefficient: {lr_simple.coef_[0]:.4f}")
    print(f"Intercept: {lr_simple.intercept_:.4f}")
    print(f"MSE: {mse_simple:.4f}")
    print(f"RMSE: {rmse_simple:.4f}")
    print(f"MAE: {mae_simple:.4f}")
    print(f"R² Score: {r2_simple:.4f}")
    
    # Sauvegarde du modèle
    joblib.dump(lr_simple, 'models/regression/linear_regression_simple.pkl')
    print(f"✓ Modèle sauvegardé: 'models/regression/linear_regression_simple.pkl'")
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[simple_feature], y_test_reg, alpha=0.5, label='Données réelles')
    plt.scatter(X_test[simple_feature], y_pred_simple, alpha=0.5, label='Prédictions', color='red')
    plt.xlabel(simple_feature)
    plt.ylabel('Score de Santé Mentale')
    plt.title(f'Régression Linéaire Simple - {simple_feature}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/05_linear_regression_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5.2 Régression Linéaire Multiple
    print("\n--- 5.2 Régression Linéaire Multiple ---")
    
    lr_multiple = LinearRegression()
    lr_multiple.fit(X_train, y_train_reg)
    y_pred_multiple = lr_multiple.predict(X_test)
    
    # Métriques
    mse_multiple = mean_squared_error(y_test_reg, y_pred_multiple)
    rmse_multiple = np.sqrt(mse_multiple)
    mae_multiple = mean_absolute_error(y_test_reg, y_pred_multiple)
    r2_multiple = r2_score(y_test_reg, y_pred_multiple)
    
    print(f"MSE: {mse_multiple:.4f}")
    print(f"RMSE: {rmse_multiple:.4f}")
    print(f"MAE: {mae_multiple:.4f}")
    print(f"R² Score: {r2_multiple:.4f}")
    
    # Sauvegarde du modèle
    joblib.dump(lr_multiple, 'models/regression/linear_regression_multiple.pkl')
    print(f"✓ Modèle sauvegardé: 'models/regression/linear_regression_multiple.pkl'")
    
    # Importance des features
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': lr_multiple.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("\nImportance des features:")
    print(feature_importance)
    
    # Visualisation des résidus
    residuals = y_test_reg - y_pred_multiple
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].scatter(y_pred_multiple, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Valeurs prédites')
    axes[0].set_ylabel('Résidus')
    axes[0].set_title('Graphique des résidus')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_test_reg, y_pred_multiple, alpha=0.5)
    axes[1].plot([y_test_reg.min(), y_test_reg.max()], 
                 [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
    axes[1].set_xlabel('Valeurs réelles')
    axes[1].set_ylabel('Valeurs prédites')
    axes[1].set_title('Prédictions vs Réalité')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/06_linear_regression_multiple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sauvegarde des résultats
    regression_results = {
        'simple': {
            'feature': simple_feature,
            'MSE': float(mse_simple),
            'RMSE': float(rmse_simple),
            'MAE': float(mae_simple),
            'R2': float(r2_simple)
        },
        'multiple': {
            'MSE': float(mse_multiple),
            'RMSE': float(rmse_multiple),
            'MAE': float(mae_multiple),
            'R2': float(r2_multiple),
            'feature_importance': feature_importance.to_dict('records')
        }
    }
    
    import json
    with open('results/regression_results.json', 'w') as f:
        json.dump(regression_results, f, indent=4)
    
    print("\n✓ Résultats de régression sauvegardés dans 'results/regression_results.json'")
    print("\n--- Partie 2 complétée: Régression Linéaire ---\n")
else:
    print("⚠ Pas de variable cible pour la régression disponible")


# =============================================================================
# PARTIE 6: RÉGRESSION POLYNOMIALE
# =============================================================================
print("=" * 80)
print("PARTIE 6: RÉGRESSION POLYNOMIALE")
print("=" * 80)

if y_regression is not None:
    # Test de différents degrés polynomiaux
    degrees = [1, 2, 3, 4]
    results_poly = []
    poly_models = {}
    
    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        lr_poly = LinearRegression()
        lr_poly.fit(X_train_poly, y_train_reg)
        y_pred_poly = lr_poly.predict(X_test_poly)
        
        mse = mean_squared_error(y_test_reg, y_pred_poly)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_reg, y_pred_poly)
        r2 = r2_score(y_test_reg, y_pred_poly)
        
        results_poly.append({
            'Degré': degree,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })
        
        # Sauvegarde du modèle et du transformer
        poly_models[degree] = {'poly': poly, 'model': lr_poly}
        joblib.dump(poly, f'models/regression/poly_features_degree_{degree}.pkl')
        joblib.dump(lr_poly, f'models/regression/poly_regression_degree_{degree}.pkl')
        
        print(f"Degré {degree}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    # Comparaison des modèles
    results_df = pd.DataFrame(results_poly)
    print("\n--- Comparaison des modèles polynomiaux ---")
    print(results_df)
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(results_df['Degré'], results_df['MSE'], marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Degré polynomial')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE vs Degré polynomial')
    axes[0].grid(True)
    
    axes[1].plot(results_df['Degré'], results_df['RMSE'], marker='o', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Degré polynomial')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE vs Degré polynomial')
    axes[1].grid(True)
    
    axes[2].plot(results_df['Degré'], results_df['R²'], marker='o', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Degré polynomial')
    axes[2].set_ylabel('R² Score')
    axes[2].set_title('R² Score vs Degré polynomial')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/07_polynomial_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sauvegarde des résultats
    with open('results/polynomial_results.json', 'w') as f:
        json.dump(results_poly, f, indent=4)
    
    print("\n✓ Modèles polynomiaux sauvegardés dans 'models/regression/'")
    print("✓ Résultats sauvegardés dans 'results/polynomial_results.json'")
    print("\n--- Partie 3 complétée: Régression Polynomiale ---\n")
else:
    print("⚠ Pas de variable cible pour la régression disponible")

# =============================================================================
# PARTIE 7: MÉTHODES DE CLASSIFICATION - KNN
# =============================================================================
print("=" * 80)
print("PARTIE 7: K-NEAREST NEIGHBORS (KNN)")
print("=" * 80)

if y_classification is not None:
    # Division des données pour la classification
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_scaled, y_classification, test_size=0.2, random_state=42, stratify=y_classification
    )
    
    # Recherche du meilleur k
    k_values = range(1, 21)
    k_scores = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_clf, y_train_clf, cv=5)
        k_scores.append(scores.mean())
    
    best_k = k_values[np.argmax(k_scores)]
    print(f"Meilleur k: {best_k} (Accuracy: {max(k_scores):.4f})")
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, k_scores, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Nombre de voisins (k)')
    plt.ylabel('Accuracy (CV)')
    plt.title('Performance du KNN en fonction de k')
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Meilleur k={best_k}')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/08_knn_k_selection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Entraînement avec le meilleur k
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train_clf, y_train_clf)
    y_pred_knn = knn_best.predict(X_test_clf)
    y_pred_proba_knn = knn_best.predict_proba(X_test_clf)[:, 1]
    
    # Sauvegarde du modèle
    joblib.dump(knn_best, 'models/classification/knn_classifier.pkl')
    print(f"✓ Modèle KNN sauvegardé: 'models/classification/knn_classifier.pkl'")
    
    # Évaluation
    print("\n--- Métriques de performance ---")
    print(f"Accuracy: {accuracy_score(y_test_clf, y_pred_knn):.4f}")
    print(f"F1-Score: {f1_score(y_test_clf, y_pred_knn):.4f}")
    print(f"\n{classification_report(y_test_clf, y_pred_knn, target_names=['Pas de problème', 'Problème de santé mentale'])}")
    
    # Matrice de confusion
    cm_knn = confusion_matrix(y_test_clf, y_pred_knn)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matrice de Confusion - KNN')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.savefig('visualizations/09_knn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Courbe ROC
    fpr_knn, tpr_knn, _ = roc_curve(y_test_clf, y_pred_proba_knn)
    roc_auc_knn = auc(fpr_knn, tpr_knn)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_knn, tpr_knn, linewidth=2, label=f'KNN (AUC = {roc_auc_knn:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC - KNN')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/10_knn_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n--- Partie 4 complétée: KNN ---\n")
else:
    print("⚠ Pas de variable cible pour la classification disponible")
