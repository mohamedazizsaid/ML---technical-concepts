"""
=============================================================================
PROJET MACHINE LEARNING APPLIQUÉ - MODULE SI-17
=============================================================================
Dataset: Student Mental Health
Objectif: Couvrir tous les acquis d'apprentissage (AA1-AA6)
Auteur: Projet académique
Date: 2024
=============================================================================
"""

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
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
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

# Note: Téléchargez le fichier CSV depuis Kaggle et placez-le dans le dossier 'data'
# URL: https://www.kaggle.com/datasets/shariful07/student-mental-health

try:
    df = pd.read_csv('data/Student Mental health.csv')
    print("✓ Dataset chargé depuis le fichier local")
except FileNotFoundError:
    print("⚠ Fichier non trouvé. Création d'un dataset d'exemple...")
   
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
# =============================================================================
# 4.1 Normalisation et Vérification de Data Leakage 
# =============================================================================
print("\n--- 4.1 Normalisation et Vérification de Data Leakage ---\n")

# IMPORTANT: Vérifier d'abord si les données corrigées existent
import os

corrected_data_exists = (
    os.path.exists('data/X_scaled_corrected.csv') and
    os.path.exists('data/y_classification_corrected.csv') and
    os.path.exists('data/y_regression_corrected.csv')
)

if corrected_data_exists:
    print("✓ Chargement des données corrigées (sans data leakage)")
    
    # Charger les données déjà corrigées et normalisées
    X_scaled = pd.read_csv('data/X_scaled_corrected.csv')
    y_classification = pd.read_csv('data/y_classification_corrected.csv').values.ravel()
    y_regression = pd.read_csv('data/y_regression_corrected.csv').values.ravel()
    
    # Charger le scaler
    scaler = joblib.load('models/scaler_corrected.pkl')
    
    # Recréer X (non normalisé) pour compatibilité
    X = X_scaled.copy()
    
    print(f"✓ X_scaled shape: {X_scaled.shape}")
    print(f"✓ y_classification shape: {y_classification.shape}")
    print(f"✓ y_regression shape: {y_regression.shape}")
    print("\n✓ Données corrigées chargées avec succès!")
    
else:
    print("⚠️ ATTENTION: Données corrigées non trouvées!")
    print("\n--- Procédure de correction manuelle ---")
    
    # Identifier les colonnes à EXCLURE (contiennent des infos sur la cible)
    columns_to_exclude = [
        'Do you have Depression?',
        'Do you have Anxiety?',
        'Do you have Panic attack?',
        'Did you seek any specialist for a treatment?',
        'Do you have Depression?_encoded',
        'Do you have Anxiety?_encoded',
        'Do you have Panic attack?_encoded',
        'Did you seek any specialist for a treatment?_encoded',
        'mental_health_issue',
        'mental_health_score',
        'Timestamp',
        'Cluster'
    ]
    
    print(f"\nColonnes à exclure ({len(columns_to_exclude)}):")
    for col in columns_to_exclude:
        if col in df_clean.columns:
            print(f"  - {col}")
    
    # Créer la liste des features VALIDES (sans data leakage)
    valid_features = []
    
    # Features numériques de base
    if 'Age' in df_clean.columns:
        valid_features.append('Age')
    
    if 'CGPA_numeric' in df_clean.columns:
        valid_features.append('CGPA_numeric')
    elif 'What is your CGPA?' in df_clean.columns:
        # Convertir CGPA si nécessaire
        import re
        def convert_cgpa(cgpa_value):
            if isinstance(cgpa_value, (int, float)):
                return float(cgpa_value)
            elif isinstance(cgpa_value, str):
                match = re.search(r'(\d+\.\d+)', str(cgpa_value))
                if match:
                    return float(match.group(1))
            return np.nan
        
        df_clean['CGPA_numeric'] = df_clean['What is your CGPA?'].apply(convert_cgpa)
        valid_features.append('CGPA_numeric')
    
    # Features encodées SÛRES (pas liées à la santé mentale)
    safe_encoded = [
        'Choose your gender_encoded',
        'Your current year of Study_encoded',
        'What is your course?_encoded',
        'Marital status_encoded'
    ]
    
    for col in safe_encoded:
        if col in df_clean.columns:
            valid_features.append(col)
    
    print(f"\n✓ Features valides identifiées ({len(valid_features)}):")
    for feat in valid_features:
        print(f"  + {feat}")
    
    # Créer X propre (sans data leakage)
    X = df_clean[valid_features].copy()
    X = X.fillna(X.median())
    
    # Normaliser
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=valid_features)
    
    # Recréer les variables cibles
    y_classification = df_clean['mental_health_issue'].copy()
    y_regression = df_clean['mental_health_score'].copy()
    
    # Vérification de data leakage
    print("\n--- Vérification de data leakage ---")
    correlations = []
    for col in X.columns:
        corr = X[col].corr(y_classification)
        correlations.append({'Feature': col, 'Correlation': abs(corr)})
    
    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
    print("\nCorrélations avec la cible:")
    print(corr_df.to_string(index=False))
    
    suspicious = corr_df[corr_df['Correlation'] > 0.95]
    if len(suspicious) > 0:
        print("\n⚠️ ATTENTION: Corrélations suspectes détectées!")
        print(suspicious)
    else:
        print("\n✓ Aucune corrélation suspecte (toutes < 0.95)")
    
    # Sauvegarder les données corrigées
    print("\n--- Sauvegarde des données corrigées ---")
    X_scaled.to_csv('data/X_scaled_corrected.csv', index=False)
    pd.DataFrame(y_classification).to_csv('data/y_classification_corrected.csv', index=False)
    pd.DataFrame(y_regression).to_csv('data/y_regression_corrected.csv', index=False)
    joblib.dump(scaler, 'models/scaler_corrected.pkl')
    
    print("✓ Données corrigées sauvegardées:")
    print("  - data/X_scaled_corrected.csv")
    print("  - data/y_classification_corrected.csv")
    print("  - data/y_regression_corrected.csv")
    print("  - models/scaler_corrected.pkl")

# Test rapide de validation
print("\n--- Test de validation rapide ---")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

X_train_test, X_test_test, y_train_test, y_test_test = train_test_split(
    X_scaled, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)

rf_test = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
cv_scores = cross_val_score(rf_test, X_train_test, y_train_test, cv=3)
rf_test.fit(X_train_test, y_train_test)

acc_train_test = rf_test.score(X_train_test, y_train_test)
acc_test_test = rf_test.score(X_test_test, y_test_test)

print(f"Accuracy Train: {acc_train_test:.4f}")
print(f"Accuracy Test:  {acc_test_test:.4f}")
print(f"Accuracy CV:    {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"Écart:          {acc_train_test - acc_test_test:.4f}")

# Diagnostic
if acc_test_test >= 0.99:
    print("\n⚠️ PROBLÈME: Accuracy trop élevée (>= 0.99)")
    print("   → Il reste du data leakage")
    print("   → Vérifiez les features utilisées")
elif acc_test_test >= 0.75:
    print("\n✅ SUCCÈS: Accuracy réaliste (0.75-0.99)")
    print("   → Pas de data leakage évident")
    print("   → Modèle prêt pour l'entraînement")
else:
    print("\n⚠️ Performance faible (< 0.75)")
    print("   → Vérifiez que vous avez assez de features pertinentes")

if acc_train_test - acc_test_test > 0.15:
    print("\n⚠️ Overfitting détecté (écart > 0.15)")
else:
    print("\n✓ Pas d'overfitting majeur (écart <= 0.15)")

print("\n✓ Données prêtes pour la modélisation!")
print(f"✓ Shape finale: X={X_scaled.shape}, y_class={y_classification.shape}, y_reg={y_regression.shape}")
print("\n--- Partie 4 complétée ---\n")


# =============================================================================
# ÉTAPE 4.2: VÉRIFICATION DE DATA LEAKAGE
# =============================================================================

print("=" * 80)
print("PARTIE 4.2: VÉRIFICATION DE DATA LEAKAGE ")
print("=" * 80)

# Fonction à ajouter
def check_data_leakage(df, feature_cols, target_col):
    """Vérifie s'il y a des fuites de données"""
    print("=== VÉRIFICATION DE DATA LEAKAGE ===\n")
    
    correlations = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_col]):
            corr = df[col].corr(df[target_col])
            correlations[col] = abs(corr)
    
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 5 features les plus corrélées avec la cible:")
    for feat, corr in sorted_corr[:5]:
        status = "⚠️ SUSPECT (>0.95)" if corr > 0.95 else "✓ OK"
        print(f"{feat}: {corr:.4f} {status}")
    
    suspicious = [feat for feat, corr in sorted_corr if corr > 0.95]
    if suspicious:
        print(f"\n⚠️ ATTENTION: Features à exclure: {suspicious}")
        return suspicious
    else:
        print("\n✓ Aucune fuite de données détectée")
        return []

# UTILISATION IMMÉDIATE
if y_classification is not None:
    print("\n--- Vérification pour la classification ---")
    suspicious_features = check_data_leakage(df_clean, feature_columns, 'mental_health_issue')
    
    # CORRECTION: Retirer les features suspectes
    if suspicious_features:
        print(f"\nRetrait des features suspectes: {suspicious_features}")
        feature_columns_clean = [col for col in feature_columns if col not in suspicious_features]
        
        # Recréer X sans les features problématiques
        X_clean = df_clean[feature_columns_clean]
        
        # Re-normaliser
        scaler_clean = StandardScaler()
        X_scaled_clean = scaler_clean.fit_transform(X_clean)
        X_scaled_clean = pd.DataFrame(X_scaled_clean, columns=feature_columns_clean)
        
        print(f"✓ Nouvelles dimensions: {X_scaled_clean.shape}")
        
        # Utiliser X_scaled_clean pour la suite
        X_scaled = X_scaled_clean
        X = X_clean
    else:
        print("✓ Pas de corrections nécessaires")

print("\n--- Partie 4.8 complétée ---\n")


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

print("\n--- 7.1 Validation robuste du KNN ---")

if y_classification is not None:
    # Test avec plusieurs random states
    print("Test de stabilité avec différents splits:")
    
    stability_scores = []
    for seed in [42, 123, 456, 789, 999]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y_classification, test_size=0.2, 
            random_state=seed, stratify=y_classification
        )
        
        knn_temp = KNeighborsClassifier(n_neighbors=best_k)
        knn_temp.fit(X_tr, y_tr)
        score = knn_temp.score(X_te, y_te)
        stability_scores.append(score)
        print(f"  Seed {seed}: Accuracy = {score:.4f}")
    
    mean_score = np.mean(stability_scores)
    std_score = np.std(stability_scores)
    
    print(f"\nMoyenne: {mean_score:.4f} (±{std_score:.4f})")
    
    if std_score < 0.05:
        print("✓ Modèle STABLE")
    else:
        print("⚠️ Modèle INSTABLE - Résultats variables selon le split")
    
    # Vérification d'overfitting
    knn_best.fit(X_train_clf, y_train_clf)
    acc_train = knn_best.score(X_train_clf, y_train_clf)
    acc_test = knn_best.score(X_test_clf, y_test_clf)
    
    print(f"\nVérification overfitting:")
    print(f"  Accuracy Train: {acc_train:.4f}")
    print(f"  Accuracy Test: {acc_test:.4f}")
    print(f"  Écart: {acc_train - acc_test:.4f}")
    
    if acc_train - acc_test > 0.1:
        print("  ⚠️ Possible overfitting détecté")
    else:
        print("  ✓ Pas d'overfitting majeur")

# =============================================================================
# PARTIE 8: SUPPORT VECTOR MACHINE (SVM)
# =============================================================================
print("=" * 80)
print("PARTIE 8: SUPPORT VECTOR MACHINE (SVM)")
print("=" * 80)

if y_classification is not None:
    # Test de différents noyaux
    kernels = ['linear', 'rbf', 'poly']
    svm_results = {}
    
    for kernel in kernels:
        print(f"\n--- SVM avec noyau {kernel.upper()} ---")
        svm = SVC(kernel=kernel, probability=True, random_state=42)
        svm.fit(X_train_clf, y_train_clf)
        y_pred_svm = svm.predict(X_test_clf)
        y_pred_proba_svm = svm.predict_proba(X_test_clf)[:, 1]
        
        acc = accuracy_score(y_test_clf, y_pred_svm)
        f1 = f1_score(y_test_clf, y_pred_svm)
        
        fpr, tpr, _ = roc_curve(y_test_clf, y_pred_proba_svm)
        roc_auc = auc(fpr, tpr)
        
        svm_results[kernel] = {
            'model': svm,
            'accuracy': acc,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'predictions': y_pred_svm
        }
        
        # Sauvegarde du modèle
        joblib.dump(svm, f'models/classification/svm_{kernel}.pkl')
        
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"✓ Modèle sauvegardé: 'models/classification/svm_{kernel}.pkl'")
    
    # Comparaison des noyaux
    comparison_df = pd.DataFrame({
        'Noyau': kernels,
        'Accuracy': [svm_results[k]['accuracy'] for k in kernels],
        'F1-Score': [svm_results[k]['f1_score'] for k in kernels],
        'ROC AUC': [svm_results[k]['roc_auc'] for k in kernels]
    })
    
    print("\n--- Comparaison des noyaux SVM ---")
    print(comparison_df)
    
    # Visualisation des courbes ROC
    plt.figure(figsize=(10, 6))
    for kernel in kernels:
        plt.plot(svm_results[kernel]['fpr'], svm_results[kernel]['tpr'], 
                 linewidth=2, label=f'{kernel.upper()} (AUC={svm_results[kernel]["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbes ROC - Comparaison des noyaux SVM')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/11_svm_roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Matrice de confusion pour le meilleur modèle
    best_kernel = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Noyau']
    print(f"\nMeilleur noyau: {best_kernel.upper()}")
    
    cm_svm = confusion_matrix(y_test_clf, svm_results[best_kernel]['predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title(f'Matrice de Confusion - SVM ({best_kernel.upper()})')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.savefig('visualizations/12_svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n--- Partie 5 complétée: SVM ---\n")
else:
    print("⚠ Pas de variable cible pour la classification disponible")

# =============================================================================
# PARTIE 9: ARBRE DE DÉCISION ET XGBOOST
# =============================================================================
print("=" * 80)
print("PARTIE 9: ARBRE DE DÉCISION ET XGBOOST")
print("=" * 80)

if y_classification is not None:
    # 9.1 Arbre de Décision
    print("--- 9.1 Arbre de Décision ---")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train_clf, y_train_clf)
    y_pred_dt = dt.predict(X_test_clf)
    y_pred_proba_dt = dt.predict_proba(X_test_clf)[:, 1]
    
    acc_dt = accuracy_score(y_test_clf, y_pred_dt)
    f1_dt = f1_score(y_test_clf, y_pred_dt)
    
    print(f"Accuracy: {acc_dt:.4f}")
    print(f"F1-Score: {f1_dt:.4f}")
    
    # Sauvegarde du modèle
    joblib.dump(dt, 'models/classification/decision_tree.pkl')
    print(f"✓ Modèle sauvegardé: 'models/classification/decision_tree.pkl'")
    
    # Importance des features
    feature_imp_dt = pd.DataFrame({
        'Feature': X.columns,
        'Importance': dt.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nImportance des features (Arbre de Décision):")
    print(feature_imp_dt.head(10))
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.barh(feature_imp_dt['Feature'][:10], feature_imp_dt['Importance'][:10])
    plt.xlabel('Importance')
    plt.title('Top 10 Features - Arbre de Décision')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('visualizations/13_decision_tree_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 9.2 Random Forest
    print("\n--- 9.2 Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train_clf, y_train_clf)
    y_pred_rf = rf.predict(X_test_clf)
    y_pred_proba_rf = rf.predict_proba(X_test_clf)[:, 1]
    
    acc_rf = accuracy_score(y_test_clf, y_pred_rf)
    f1_rf = f1_score(y_test_clf, y_pred_rf)
    
    print(f"Accuracy: {acc_rf:.4f}")
    print(f"F1-Score: {f1_rf:.4f}")
    
    # Sauvegarde du modèle
    joblib.dump(rf, 'models/classification/random_forest.pkl')
    print(f"✓ Modèle sauvegardé: 'models/classification/random_forest.pkl'")
    
    # 9.3 XGBoost
    print("\n--- 9.3 XGBoost ---")
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                        random_state=42, eval_metric='logloss')
    xgb.fit(X_train_clf, y_train_clf)
    y_pred_xgb = xgb.predict(X_test_clf)
    y_pred_proba_xgb = xgb.predict_proba(X_test_clf)[:, 1]
    
    acc_xgb = accuracy_score(y_test_clf, y_pred_xgb)
    f1_xgb = f1_score(y_test_clf, y_pred_xgb)
    
    print(f"Accuracy: {acc_xgb:.4f}")
    print(f"F1-Score: {f1_xgb:.4f}")
    
    # Sauvegarde du modèle
    joblib.dump(xgb, 'models/classification/xgboost.pkl')
    print(f"✓ Modèle sauvegardé: 'models/classification/xgboost.pkl'")
    
    # Importance des features
    feature_imp_xgb = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nImportance des features (XGBoost):")
    print(feature_imp_xgb.head(10))
    
    # Matrice de confusion XGBoost
    cm_xgb = confusion_matrix(y_test_clf, y_pred_xgb)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', cbar=False)
    plt.title('Matrice de Confusion - XGBoost')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.savefig('visualizations/15_xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n--- Partie 9.1-9.3 complétée ---\n")
else:
    print("⚠ Pas de variable cible pour la classification disponible")

# =============================================================================
# PARTIE 9.4: NAIVE BAYES
# =============================================================================
print("=" * 80)
print("PARTIE 9.4: NAIVE BAYES")
print("=" * 80)

if y_classification is not None:
    print("--- Naive Bayes ---")
    # Initialisation et entraînement du modèle Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_clf, y_train_clf)
    y_pred_nb = nb.predict(X_test_clf)
    y_pred_proba_nb = nb.predict_proba(X_test_clf)[:, 1]
    
    # Calcul des métriques
    acc_nb = accuracy_score(y_test_clf, y_pred_nb)
    f1_nb = f1_score(y_test_clf, y_pred_nb)
    
    print(f"Accuracy: {acc_nb:.4f}")
    print(f"F1-Score: {f1_nb:.4f}")
    
    # Sauvegarde du modèle
    joblib.dump(nb, 'models/classification/naive_bayes.pkl')
    print(f"✓ Modèle sauvegardé: 'models/classification/naive_bayes.pkl'")
    
    # Matrice de confusion
    cm_nb = confusion_matrix(y_test_clf, y_pred_nb)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Purples', cbar=False)
    plt.title('Matrice de Confusion - Naive Bayes')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.savefig('visualizations/16_naive_bayes_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Courbe ROC
    fpr_nb, tpr_nb, _ = roc_curve(y_test_clf, y_pred_proba_nb)
    roc_auc_nb = auc(fpr_nb, tpr_nb)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_nb, tpr_nb, linewidth=2, label=f'Naive Bayes (AUC = {roc_auc_nb:.3f})', color='purple')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC - Naive Bayes')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/17_naive_bayes_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n--- Partie 9.4 complétée ---\n")

# =============================================================================
# PARTIE 9.5: GRADIENT BOOSTING
# =============================================================================
print("=" * 80)
print("PARTIE 9.5: GRADIENT BOOSTING")
print("=" * 80)

if y_classification is not None:
    print("--- Gradient Boosting ---")
    # Initialisation et entraînement du modèle Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        subsample=0.8
    )
    gb.fit(X_train_clf, y_train_clf)
    y_pred_gb = gb.predict(X_test_clf)
    y_pred_proba_gb = gb.predict_proba(X_test_clf)[:, 1]
    
    # Calcul des métriques
    acc_gb = accuracy_score(y_test_clf, y_pred_gb)
    f1_gb = f1_score(y_test_clf, y_pred_gb)
    
    print(f"Accuracy: {acc_gb:.4f}")
    print(f"F1-Score: {f1_gb:.4f}")
    
    # Sauvegarde du modèle
    joblib.dump(gb, 'models/classification/gradient_boosting.pkl')
    print(f"✓ Modèle sauvegardé: 'models/classification/gradient_boosting.pkl'")
    
    # Importance des features
    feature_imp_gb = pd.DataFrame({
        'Feature': X.columns,
        'Importance': gb.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nImportance des features (Gradient Boosting):")
    print(feature_imp_gb.head(10))
    
    # Visualisation de l'importance des features
    plt.figure(figsize=(10, 6))
    plt.barh(feature_imp_gb['Feature'][:10], feature_imp_gb['Importance'][:10], color='teal')
    plt.xlabel('Importance')
    plt.title('Top 10 Features - Gradient Boosting')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('visualizations/18_gradient_boosting_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Matrice de confusion
    cm_gb = confusion_matrix(y_test_clf, y_pred_gb)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_gb, annot=True, fmt='d', cmap='YlOrBr', cbar=False)
    plt.title('Matrice de Confusion - Gradient Boosting')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.savefig('visualizations/19_gradient_boosting_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Courbe ROC
    fpr_gb, tpr_gb, _ = roc_curve(y_test_clf, y_pred_proba_gb)
    roc_auc_gb = auc(fpr_gb, tpr_gb)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_gb, tpr_gb, linewidth=2, label=f'Gradient Boosting (AUC = {roc_auc_gb:.3f})', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC - Gradient Boosting')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/20_gradient_boosting_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Apprentissage de la courbe d'erreur
    train_scores = []
    test_scores = []
    
    for n_estimators in range(1, 101, 10):
        gb_temp = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_temp.fit(X_train_clf, y_train_clf)
        
        train_scores.append(gb_temp.score(X_train_clf, y_train_clf))
        test_scores.append(gb_temp.score(X_test_clf, y_test_clf))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 101, 10), train_scores, label='Score entraînement', marker='o', linewidth=2)
    plt.plot(range(1, 101, 10), test_scores, label='Score test', marker='s', linewidth=2)
    plt.xlabel('Nombre d\'estimateurs')
    plt.ylabel('Accuracy')
    plt.title('Courbe d\'apprentissage - Gradient Boosting')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/21_gradient_boosting_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n--- Partie 9.5 complétée ---\n")

# =============================================================================
# PARTIE 9.6: COMPARAISON DE TOUS LES MODÈLES
# =============================================================================
print("=" * 80)
print("PARTIE 9.6: COMPARAISON DE TOUS LES MODÈLES DE CLASSIFICATION")
print("=" * 80)

if y_classification is not None:
    # Comparaison des modèles (MAINTENANT toutes les variables sont définies)
    models_comparison = pd.DataFrame({
        'Modèle': ['KNN', f'SVM ({best_kernel})', 'Decision Tree', 'Random Forest', 'XGBoost', 'Naive Bayes', 'Gradient Boosting'],
        'Accuracy': [
            accuracy_score(y_test_clf, y_pred_knn),
            svm_results[best_kernel]['accuracy'],
            acc_dt,
            acc_rf,
            acc_xgb,
            acc_nb,  
            acc_gb   
        ],
        'F1-Score': [
            f1_score(y_test_clf, y_pred_knn),
            svm_results[best_kernel]['f1_score'],
            f1_dt,
            f1_rf,
            f1_xgb,
            f1_nb,   
            f1_gb    
        ]
    })
    
    print("\n--- Comparaison de tous les modèles de classification ---")
    print(models_comparison)
    
    # Visualisation comparative
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    models_comparison.plot(x='Modèle', y='Accuracy', kind='bar', ax=axes[0], legend=False, color='steelblue')
    axes[0].set_title('Comparaison des Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0.5, 1.0])
    axes[0].grid(axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    models_comparison.plot(x='Modèle', y='F1-Score', kind='bar', ax=axes[1], 
                           legend=False, color='coral')
    axes[1].set_title('Comparaison des F1-Scores')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_ylim([0.5, 1.0])
    axes[1].grid(axis='y')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/22_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sauvegarde des résultats
    classification_results = models_comparison.to_dict('records')
    with open('results/classification_results.json', 'w') as f:
        json.dump(classification_results, f, indent=4)
    
    print("\n✓ Résultats de classification sauvegardés dans 'results/classification_results.json'")
    print("\n--- Partie 9 complétée: Tous les modèles de classification ---\n")
# =============================================================================
# PARTIE 10: CLUSTERING - K-MEANS
# =============================================================================
print("=" * 80)
print("PARTIE 10: CLUSTERING - K-MEANS")
print("=" * 80)

# Méthode du coude pour déterminer le nombre optimal de clusters
print("--- Méthode du coude (Elbow Method) ---")
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
axes[0].set_xlabel('Nombre de clusters (k)')
axes[0].set_ylabel('Inertie')
axes[0].set_title('Méthode du coude')
axes[0].grid(True)

axes[1].plot(K_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='green')
axes[1].set_xlabel('Nombre de clusters (k)')
axes[1].set_ylabel('Score de Silhouette')
axes[1].set_title('Score de Silhouette vs k')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('visualizations/16_kmeans_elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()

# Choix du nombre optimal de clusters
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nNombre optimal de clusters: {optimal_k}")

# Application du K-Means avec k optimal
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_optimal.fit_predict(X_scaled)

# Sauvegarde du modèle
joblib.dump(kmeans_optimal, 'models/clustering/kmeans_optimal.pkl')
print(f"✓ Modèle K-Means sauvegardé: 'models/clustering/kmeans_optimal.pkl'")

# Ajout des clusters au dataframe
df_clean['Cluster'] = clusters

# Statistiques par cluster
print(f"\n--- Distribution des clusters ---")
print(df_clean['Cluster'].value_counts().sort_index())

print("\n--- Caractéristiques moyennes par cluster ---")
# Ne prendre que les colonnes numériques
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
cluster_cols = [col for col in numeric_cols if col in ['Age', 'What is your CGPA?', 'mental_health_score', 'Cluster']]
# Exclure Cluster de la liste des colonnes à moyenner
cluster_cols = [col for col in cluster_cols if col != 'Cluster']

if cluster_cols:
    cluster_summary = df_clean.groupby('Cluster')[cluster_cols].mean()
    print(cluster_summary)
else:
    print("Aucune colonne numérique disponible pour l'analyse par cluster")
# Visualisation des clusters (2D avec PCA)
pca_viz = PCA(n_components=2)
X_pca_viz = pca_viz.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca_viz[:, 0], X_pca_viz[:, 1], c=clusters, 
                     cmap='viridis', s=50, alpha=0.6, edgecolors='black')

# Transformer les centroïdes dans l'espace PCA
centers_pca = pca_viz.transform(kmeans_optimal.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
           c='red', s=300, alpha=0.8, marker='X', edgecolors='black',
           linewidths=2, label='Centroïdes')

plt.xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}%)')
plt.title(f'Visualisation des Clusters K-Means (k={optimal_k})')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('visualizations/17_kmeans_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyse de la relation entre clusters et problèmes de santé mentale
if 'mental_health_issue' in df_clean.columns:
    print("\n--- Relation Clusters vs Problèmes de santé mentale ---")
    cluster_mental_health = pd.crosstab(df_clean['Cluster'], df_clean['mental_health_issue'], 
                                  normalize='index') * 100
    print(cluster_mental_health)
    
    cluster_mental_health.plot(kind='bar', stacked=False, figsize=(10, 6), color=['#2ecc71', '#e74c3c'])
    plt.title('Pourcentage de problèmes de santé mentale par cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Pourcentage (%)')
    plt.legend(['Pas de problème', 'Problème'])
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/18_clusters_vs_mental_health.png', dpi=300, bbox_inches='tight')
    plt.show()

# Métriques de performance
final_silhouette = silhouette_score(X_scaled, clusters)
print(f"\n--- Métriques finales K-Means ---")
print(f"Silhouette Score: {final_silhouette:.4f}")
print(f"Inertie: {kmeans_optimal.inertia_:.2f}")

# Sauvegarde des résultats
clustering_results = {
    'optimal_k': int(optimal_k),
    'silhouette_score': float(final_silhouette),
    'inertia': float(kmeans_optimal.inertia_),
    'cluster_distribution': df_clean['Cluster'].value_counts().to_dict()
}

with open('results/clustering_results.json', 'w') as f:
    json.dump(clustering_results, f, indent=4)

print("\n✓ Résultats de clustering sauvegardés dans 'results/clustering_results.json'")
print("\n--- Partie 7 complétée: K-Means Clustering ---\n")

# =============================================================================
# PARTIE 11: ANALYSE EN COMPOSANTES PRINCIPALES (ACP/PCA)
# =============================================================================
print("=" * 80)
print("PARTIE 11: ANALYSE EN COMPOSANTES PRINCIPALES (PCA)")
print("=" * 80)

# Application de la PCA
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# Sauvegarde du modèle PCA
joblib.dump(pca_full, 'models/dimensionality_reduction/pca_full.pkl')
print("✓ Modèle PCA sauvegardé: 'models/dimensionality_reduction/pca_full.pkl'")

# Variance expliquée
variance_explained = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_explained)

print("\n--- Variance expliquée par composante ---")
for i, (var, cum_var) in enumerate(zip(variance_explained, cumulative_variance), 1):
    print(f"PC{i}: {var*100:.2f}% (Cumulative: {cum_var*100:.2f}%)")

# Visualisation de la variance expliquée
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Variance par composante
axes[0].bar(range(1, len(variance_explained) + 1), variance_explained * 100, color='steelblue')
axes[0].set_xlabel('Composante principale')
axes[0].set_ylabel('Variance expliquée (%)')
axes[0].set_title('Variance expliquée par composante')
axes[0].grid(axis='y', alpha=0.3)

# Variance cumulée
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 
            marker='o', linewidth=2, markersize=8)
axes[1].axhline(y=95, color='r', linestyle='--', label='95% de variance')
axes[1].axhline(y=90, color='orange', linestyle='--', label='90% de variance')
axes[1].set_xlabel('Nombre de composantes')
axes[1].set_ylabel('Variance cumulée (%)')
axes[1].set_title('Variance cumulée expliquée')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/19_pca_variance_explained.png', dpi=300, bbox_inches='tight')
plt.show()

# Nombre de composantes pour 95% de variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nNombre de composantes pour 95% de variance: {n_components_95}")

# PCA avec nombre optimal de composantes
pca_optimal = PCA(n_components=n_components_95)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)

# Sauvegarde du modèle PCA optimal
joblib.dump(pca_optimal, 'models/dimensionality_reduction/pca_optimal.pkl')
print(f"✓ Modèle PCA optimal sauvegardé: 'models/dimensionality_reduction/pca_optimal.pkl'")

print(f"\nDimension originale: {X_scaled.shape[1]}")
print(f"Dimension réduite: {X_pca_optimal.shape[1]}")
print(f"Réduction dimensionnelle: {(1 - X_pca_optimal.shape[1]/X_scaled.shape[1])*100:.1f}%")

# Matrice de chargement (loadings)
loadings = pca_optimal.components_.T * np.sqrt(pca_optimal.explained_variance_)
loading_matrix = pd.DataFrame(
    loadings,
    columns=[f'PC{i+1}' for i in range(n_components_95)],
    index=X.columns
)

print("\n--- Matrice de chargement (Top 5 features par composante) ---")
for col in loading_matrix.columns[:min(3, n_components_95)]:
    print(f"\n{col}:")
    print(loading_matrix[col].abs().sort_values(ascending=False).head(5))

# Visualisation de la matrice de chargement
plt.figure(figsize=(12, 8))
sns.heatmap(loading_matrix, cmap='coolwarm', center=0, annot=True, fmt='.2f', cbar_kws={"shrink": 0.8})
plt.title('Matrice de chargement PCA')
plt.tight_layout()
plt.savefig('visualizations/20_pca_loadings.png', dpi=300, bbox_inches='tight')
plt.show()

# Biplot (2 premières composantes)
if y_classification is not None:
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca_full[:, 0], X_pca_full[:, 1], 
                         c=y_classification, cmap='RdYlBu', 
                         alpha=0.6, edgecolors='black', s=50)
    plt.xlabel(f'PC1 ({variance_explained[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({variance_explained[1]*100:.1f}% variance)')
    plt.title('Biplot - PCA (2 premières composantes)')
    plt.colorbar(scatter, label='Problème de santé mentale')
    plt.grid(True, alpha=0.3)
    
    # Ajout des vecteurs de features
    scale_factor = 3
    for i, feature in enumerate(X.columns[:min(10, len(X.columns))]):  # Limiter à 10 features
        plt.arrow(0, 0, 
                 pca_full.components_[0, i] * scale_factor,
                 pca_full.components_[1, i] * scale_factor,
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)
        plt.text(pca_full.components_[0, i] * scale_factor * 1.15,
                pca_full.components_[1, i] * scale_factor * 1.15,
                feature[:10], fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/21_pca_biplot.png', dpi=300, bbox_inches='tight')
    plt.show()

# Application de la PCA pour améliorer les modèles
if y_classification is not None:
    print("\n--- Test des modèles avec PCA ---")
    
    # Division des données PCA
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_pca_optimal, y_classification, test_size=0.2, random_state=42, stratify=y_classification
    )
    
    # Test avec différents modèles
    models_pca = {
        'KNN': KNeighborsClassifier(n_neighbors=best_k),
        'SVM': SVC(kernel=best_kernel, probability=True, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
    }
    
    pca_results = []
    for name, model in models_pca.items():
        model.fit(X_train_pca, y_train_pca)
        y_pred = model.predict(X_test_pca)
        
        acc = accuracy_score(y_test_pca, y_pred)
        f1 = f1_score(y_test_pca, y_pred)
        
        pca_results.append({
            'Modèle': name,
            'Accuracy': acc,
            'F1-Score': f1
        })
        
        # Sauvegarde du modèle avec PCA
        joblib.dump(model, f'models/dimensionality_reduction/{name.lower()}_with_pca.pkl')
        
        print(f"{name}: Accuracy={acc:.4f}, F1-Score={f1:.4f}")
    
    # Comparaison avec/sans PCA
    pca_comparison = pd.DataFrame(pca_results)
    print("\n--- Comparaison des performances avec PCA ---")
    print(pca_comparison)
    
    # Sauvegarde des résultats
    pca_results_dict = {
        'n_components': int(n_components_95),
        'variance_explained': float(cumulative_variance[n_components_95-1]),
        'dimension_reduction': float((1 - X_pca_optimal.shape[1]/X_scaled.shape[1])*100),
        'model_performances': pca_results
    }
    
    with open('results/pca_results.json', 'w') as f:
        json.dump(pca_results_dict, f, indent=4)
    
    print("\n✓ Résultats PCA sauvegardés dans 'results/pca_results.json'")

print("\n--- Partie 8 complétée: PCA ---\n")

# =============================================================================
# PARTIE 12: SYSTÈMES DE RECOMMANDATION
# =============================================================================
print("=" * 80)
print("PARTIE 12: SYSTÈMES DE RECOMMANDATION")
print("=" * 80)

print("""
Dans le contexte de la santé mentale des étudiants, nous allons créer un système 
de recommandation qui suggère des étudiants similaires pour l'analyse comparative 
et les groupes de soutien.
""")

# Création d'une matrice étudiant-caractéristiques
from sklearn.metrics.pairwise import cosine_similarity

# 12.1 Recommandation basée sur le contenu (Content-Based)
print("\n--- 12.1 Système de recommandation basé sur le contenu ---")

# Calcul de la similarité cosinus
similarity_matrix = cosine_similarity(X_scaled)

# Sauvegarde de la matrice de similarité (échantillon)
np.save('models/recommendation/similarity_matrix.npy', similarity_matrix)
print("✓ Matrice de similarité sauvegardée: 'models/recommendation/similarity_matrix.npy'")

def recommend_similar_students(student_idx, top_n=5):
    """
    Recommande les étudiants les plus similaires
    """
    similarities = similarity_matrix[student_idx]
    similar_indices = similarities.argsort()[::-1][1:top_n+1]  # Exclure l'étudiant lui-même
    
    recommendations = []
    for idx in similar_indices:
        rec_data = {
            'Student_ID': int(idx),
            'Similarité': float(similarities[idx])
        }
        
        if 'Age' in df_clean.columns:
            rec_data['Âge'] = int(df_clean.iloc[idx]['Age'])
        if 'mental_health_issue' in df_clean.columns:
            rec_data['Problème'] = int(df_clean.iloc[idx]['mental_health_issue'])
        if 'Cluster' in df_clean.columns:
            rec_data['Cluster'] = int(df_clean.iloc[idx]['Cluster'])
        
        recommendations.append(rec_data)
    
    return pd.DataFrame(recommendations)

# Exemple de recommandation
test_student = 10
print(f"\nCaractéristiques de l'étudiant {test_student}:")
if 'Age' in df_clean.columns:
    print(f"Âge: {df_clean.iloc[test_student]['Age']}")
if 'mental_health_issue' in df_clean.columns:
    print(f"Problème de santé mentale: {'Oui' if df_clean.iloc[test_student]['mental_health_issue'] == 1 else 'Non'}")
if 'Cluster' in df_clean.columns:
    print(f"Cluster: {df_clean.iloc[test_student]['Cluster']}")

print(f"\nTop 5 étudiants similaires à l'étudiant {test_student}:")
recommendations = recommend_similar_students(test_student)
print(recommendations)

# 12.2 Recommandation collaborative (Collaborative Filtering)
print("\n--- 12.2 Filtrage collaboratif ---")

# Création d'une matrice utilisateur-item simulée
np.random.seed(42)
n_students = len(df_clean)
n_resources = 5

# Matrice étudiant x ressource (1-5 étoiles, avec des valeurs manquantes)
resource_ratings = np.random.randint(1, 6, size=(n_students, n_resources)).astype(float)
# Ajout de valeurs manquantes (30%)
mask = np.random.random((n_students, n_resources)) < 0.3
resource_ratings[mask] = np.nan

resource_df = pd.DataFrame(
    resource_ratings,
    columns=[f'Ressource_{i+1}' for i in range(n_resources)]
)

print(f"Matrice de ratings: {resource_df.shape}")
print(f"Valeurs manquantes: {resource_df.isnull().sum().sum()}")
print("\nAperçu des ratings:")
print(resource_df.head(10))

# Sauvegarde de la matrice de ratings
resource_df.to_csv('models/recommendation/resource_ratings.csv', index=False)
print("✓ Matrice de ratings sauvegardée: 'models/recommendation/resource_ratings.csv'")

# Calcul de la similarité entre étudiants basée sur leurs ratings
resource_filled = resource_df.fillna(resource_df.mean())
student_similarity = cosine_similarity(resource_filled)

def collaborative_recommend(student_idx, k_neighbors=10, top_n=3):
    """
    Recommande des ressources basées sur des étudiants similaires
    """
    # Trouver les k étudiants les plus similaires
    similarities = student_similarity[student_idx]
    similar_students = similarities.argsort()[::-1][1:k_neighbors+1]
    
    # Prédire les ratings pour les ressources non évaluées
    student_ratings = resource_df.iloc[student_idx]
    unrated_resources = student_ratings[student_ratings.isnull()].index
    
    recommendations = []
    for resource in unrated_resources:
        # Moyenne pondérée des ratings des étudiants similaires
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_student in similar_students:
            if not np.isnan(resource_df.iloc[similar_student][resource]):
                weighted_sum += similarities[similar_student] * resource_df.iloc[similar_student][resource]
                similarity_sum += similarities[similar_student]
        
        if similarity_sum > 0:
            predicted_rating = weighted_sum / similarity_sum
            recommendations.append({
                'Ressource': resource,
                'Rating_prédit': predicted_rating
            })
    
    recommendations_df = pd.DataFrame(recommendations)
    if not recommendations_df.empty:
        return recommendations_df.sort_values('Rating_prédit', ascending=False).head(top_n)
    return recommendations_df

# Exemple de recommandation collaborative
print(f"\nRecommandations de ressources pour l'étudiant {test_student}:")
print(f"Ressources déjà évaluées:")
print(resource_df.iloc[test_student].dropna())

print(f"\nRessources recommandées:")
collab_recommendations = collaborative_recommend(test_student)
if not collab_recommendations.empty:
    print(collab_recommendations)
else:
    print("Aucune recommandation disponible")

# 12.3 Système hybride
print("\n--- 12.3 Système de recommandation hybride ---")

def hybrid_recommend(student_idx, alpha=0.5, top_n=5):
    """
    Combine les approches content-based et collaborative
    alpha: poids pour content-based (1-alpha pour collaborative)
    """
    # Recommandations content-based
    content_recs = recommend_similar_students(student_idx, top_n=10)
    
    # Normaliser les scores
    if not content_recs.empty and 'Similarité' in content_recs.columns:
        content_recs['Score_content'] = content_recs['Similarité'] / content_recs['Similarité'].max()
        
        # Pour simplifier, utilisons la similarité du cluster comme score collaboratif
        if 'Cluster' in df_clean.columns:
            student_cluster = df_clean.iloc[student_idx]['Cluster']
            content_recs['Score_collab'] = content_recs['Student_ID'].apply(
                lambda x: 1.0 if df_clean.iloc[x]['Cluster'] == student_cluster else 0.5
            )
        else:
            content_recs['Score_collab'] = 0.5
        
        # Score hybride
        content_recs['Score_hybride'] = (alpha * content_recs['Score_content'] + 
                                         (1 - alpha) * content_recs['Score_collab'])
        
        return content_recs.sort_values('Score_hybride', ascending=False).head(top_n)
    
    return content_recs

print(f"\nRecommandations hybrides pour l'étudiant {test_student}:")
hybrid_recs = hybrid_recommend(test_student)
print(hybrid_recs)

# Visualisation du système de recommandation
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Heatmap de similarité (échantillon)
sample_size = min(30, len(df_clean))
sample_similarity = similarity_matrix[:sample_size, :sample_size]
sns.heatmap(sample_similarity, cmap='YlOrRd', ax=axes[0], cbar=True)
axes[0].set_title('Matrice de similarité entre étudiants (échantillon)')
axes[0].set_xlabel('Student ID')
axes[0].set_ylabel('Student ID')

# Distribution des scores de similarité
axes[1].hist(similarity_matrix[test_student], bins=30, color='skyblue', edgecolor='black')
axes[1].set_xlabel('Score de similarité')
axes[1].set_ylabel('Nombre d\'étudiants')
axes[1].set_title(f'Distribution des similarités pour l\'étudiant {test_student}')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/22_recommendation_system.png', dpi=300, bbox_inches='tight')
plt.show()

# Évaluation du système de recommandation
print("\n--- Évaluation du système de recommandation ---")

# Calcul de la précision des recommandations
if 'mental_health_issue' in df_clean.columns:
    precision_scores = []
    
    for student_idx in range(min(50, len(df_clean))):
        student_diagnosis = df_clean.iloc[student_idx]['mental_health_issue']
        similar = recommend_similar_students(student_idx, top_n=5)
        
        if not similar.empty and 'Student_ID' in similar.columns:
            similar_diagnoses = [df_clean.iloc[int(idx)]['mental_health_issue'] 
                               for idx in similar['Student_ID']]
            precision = sum([1 for d in similar_diagnoses if d == student_diagnosis]) / len(similar_diagnoses)
            precision_scores.append(precision)
    
    if precision_scores:
        mean_precision = np.mean(precision_scores)
        print(f"Précision moyenne du système: {mean_precision:.4f}")
        print(f"Cela signifie que {mean_precision*100:.1f}% des étudiants recommandés ont le même statut")
    else:
        print("Aucune évaluation disponible")

print("\n--- Partie 9 complétée: Systèmes de Recommandation ---\n")



# =============================================================================
# PARTIE 12: ANALYSE DE ROBUSTESSE DES MODÈLES
# =============================================================================

print("=" * 80)
print("PARTIE 12.5: ANALYSE DE ROBUSTESSE DES MODÈLES")
print("=" * 80)

if y_classification is not None:
    print("\n--- Test de robustesse pour tous les modèles ---\n")
    
    models_to_test = {
        'KNN': KNeighborsClassifier(n_neighbors=best_k),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
    }
    
    robustness_results = []
    
    for model_name, model in models_to_test.items():
        print(f"Test de {model_name}...")
        
        # Test avec 5 splits différents
        scores = []
        for seed in [42, 123, 456, 789, 999]:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_scaled, y_classification, test_size=0.2,
                random_state=seed, stratify=y_classification
            )
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_tr, y_tr)
            score = model_copy.score(X_te, y_te)
            scores.append(score)
        
        mean_acc = np.mean(scores)
        std_acc = np.std(scores)
        
        # Évaluation de la stabilité
        if std_acc < 0.02:
            stability = "Excellent"
        elif std_acc < 0.05:
            stability = "Bon"
        elif std_acc < 0.10:
            stability = "Moyen"
        else:
            stability = "Faible"
        
        robustness_results.append({
            'Modèle': model_name,
            'Acc Moyenne': f"{mean_acc:.4f}",
            'Écart-type': f"{std_acc:.4f}",
            'Stabilité': stability,
            'Scores': scores
        })
        
        print(f"  Moyenne: {mean_acc:.4f} ± {std_acc:.4f} [{stability}]")
    
    # Visualisation de la robustesse
    plt.figure(figsize=(12, 6))
    
    for i, result in enumerate(robustness_results):
        scores = result['Scores']
        plt.violinplot([scores], positions=[i], showmeans=True, showmedians=True)
        plt.scatter([i]*len(scores), scores, alpha=0.5, s=50)
    
    plt.xticks(range(len(robustness_results)), 
               [r['Modèle'] for r in robustness_results])
    plt.ylabel('Accuracy')
    plt.title('Robustesse des modèles (5 splits différents)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/23_model_robustness.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Analyse de robustesse complétée")
    print("✓ Le modèle le plus stable devrait être privilégié en production")

print("\n--- Analyse de robustesse complétée ---\n")


# =============================================================================
# PARTIE 13: RÉSUMÉ FINAL ET EXPORT DES RÉSULTATS
# =============================================================================
print("=" * 80)
print("PARTIE 13: RÉSUMÉ FINAL DU PROJET")
print("=" * 80)

# Résumé de tous les modèles
summary_data = []

# Régression
if y_regression is not None:
    summary_data.append(['Régression', 'Linear Simple', f'R²={r2_simple:.4f}'])
    summary_data.append(['Régression', 'Linear Multiple', f'R²={r2_multiple:.4f}'])

# Classification
if y_classification is not None:
    summary_data.append(['Classification', 'KNN', f'Acc={accuracy_score(y_test_clf, y_pred_knn):.4f}'])
    summary_data.append(['Classification', f'SVM ({best_kernel})', f'Acc={svm_results[best_kernel]["accuracy"]:.4f}'])
    summary_data.append(['Classification', 'Decision Tree', f'Acc={acc_dt:.4f}'])
    summary_data.append(['Classification', 'Random Forest', f'Acc={acc_rf:.4f}'])
    summary_data.append(['Classification', 'XGBoost', f'Acc={acc_xgb:.4f}'])
    summary_data.append(['Classification', 'Naive Bayes', f'Acc={acc_nb:.4f}'])
    summary_data.append(['Classification', 'Gradient Boosting', f'Acc={acc_gb:.4f}'])
# Clustering et PCA
summary_data.append(['Clustering', f'K-Means (k={optimal_k})', f'Silhouette={final_silhouette:.4f}'])
summary_data.append(['Réduction dim.', f'PCA ({n_components_95} comp.)', f'Variance={cumulative_variance[n_components_95-1]*100:.1f}%'])

final_summary = pd.DataFrame(summary_data, columns=['Catégorie', 'Méthode', 'Métrique principale'])

print("\n📊 TABLEAU RÉCAPITULATIF DE TOUS LES MODÈLES")
print("=" * 80)
print(final_summary.to_string(index=False))

# Meilleur modèle par catégorie
print("\n🏆 MEILLEURS MODÈLES PAR CATÉGORIE")
print("=" * 80)

if y_regression is not None:
    print(f"✓ Régression: Linear Multiple (R² = {r2_multiple:.4f})")

if y_classification is not None:
    best_clf_model = models_comparison.loc[models_comparison['Accuracy'].idxmax(), 'Modèle']
    best_clf_acc = models_comparison['Accuracy'].max()
    best_clf_f1 = models_comparison.loc[models_comparison['Accuracy'].idxmax(), 'F1-Score']
    print(f"✓ Classification: {best_clf_model} (Accuracy = {best_clf_acc:.4f}, F1-Score = {best_clf_f1:.4f})")

print(f"✓ Clustering: K-Means avec k={optimal_k} (Silhouette = {final_silhouette:.4f})")
print(f"✓ Réduction dimensionnelle: PCA à {n_components_95} composantes (95% variance)")

# Acquis d'apprentissage validés
print("\n✅ ACQUIS D'APPRENTISSAGE VALIDÉS")
print("=" * 80)
print("AA1: Identification et explication des concepts clés du ML ✓")
print("AA2: Différenciation des phases d'un projet ML ✓")
print("AA3: Préparation des données avec Python (Numpy, Pandas, Sklearn) ✓")
print("AA4: Analyse, création et évaluation de modèles de régression ✓")
print("AA5: Application et évaluation de méthodes de classification ✓")
print("AA6: Création et évaluation de modèles de segmentation ✓")

# Sauvegarde du résumé final
print("\n💾 EXPORT DES RÉSULTATS")
print("=" * 80)

# Sauvegarde du résumé en CSV
final_summary.to_csv('results/final_summary.csv', index=False)
print("✓ Résumé sauvegardé: 'results/final_summary.csv'")

# Création d'un rapport HTML complet
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Rapport ML - Student Mental Health</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .success {{ color: #27ae60; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>📊 Rapport de Projet Machine Learning</h1>
    <h2>Dataset: Student Mental Health</h2>
    
    <div class="metric">
        <h3>Informations sur le dataset</h3>
        <p><strong>Nombre d'échantillons:</strong> {len(df_clean)}</p>
        <p><strong>Nombre de features:</strong> {len(X.columns)}</p>
        <p><strong>Variables cibles:</strong> mental_health_issue, mental_health_score</p>
    </div>
    
    <h2>Résultats des Modèles</h2>
    {final_summary.to_html(index=False)}
    
    <h2 class="success">✅ Tous les Acquis d'Apprentissage Validés</h2>
    <ul>
        <li>AA1: Identification et explication des concepts clés du ML</li>
        <li>AA2: Différenciation des phases d'un projet ML</li>
        <li>AA3: Préparation des données avec Python</li>
        <li>AA4: Analyse, création et évaluation de modèles de régression</li>
        <li>AA5: Application et évaluation de méthodes de classification</li>
        <li>AA6: Création et évaluation de modèles de segmentation</li>
    </ul>
    
    <p><em>Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
</body>
</html>
"""

with open('results/final_report.html', 'w', encoding='utf-8') as f:
    f.write(html_report)

print("✓ Rapport HTML généré: 'results/final_report.html'")

# Statistiques finales
print("\n📈 STATISTIQUES FINALES")
print("=" * 80)
print(f"Total de modèles entraînés: {len(summary_data)}")
print(f"Parties complétées: 9/9 ✓")
print(f"Dataset traité: {len(df_clean)} étudiants")
print(f"Features originales: {len(X.columns)}")
print(f"Features après PCA: {n_components_95}")
print(f"Modèles sauvegardés: {len(os.listdir('models/classification')) + len(os.listdir('models/regression')) + len(os.listdir('models/clustering'))}")
print(f"Visualisations générées: {len([f for f in os.listdir('visualizations') if f.endswith('.png')])}")


print("\n" + "=" * 80)
print("🎉 PROJET MACHINE LEARNING APPLIQUÉ - TERMINÉ AVEC SUCCÈS!")
print("=" * 80)
