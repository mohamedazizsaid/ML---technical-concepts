"""
=============================================================================
CORRECTION URGENTE - DATA LEAKAGE
=============================================================================
Exécutez ce code AVANT toute modélisation
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =============================================================================
# ÉTAPE 1: CHARGER VOS DONNÉES
# =============================================================================
print("=" * 80)
print("CORRECTION DU DATA LEAKAGE")
print("=" * 80)

# Chargez votre dataset nettoyé
df_clean = pd.read_csv('data/student_mental_health_cleaned.csv')

print(f"\nDataset chargé: {df_clean.shape}")
print(f"Colonnes: {list(df_clean.columns)}\n")

# =============================================================================
# ÉTAPE 2: IDENTIFIER LES COLONNES PROBLÉMATIQUES
# =============================================================================
print("--- Identification des colonnes à EXCLURE ---\n")

# Ces colonnes doivent être EXCLUES car elles révèlent directement la cible
columns_to_exclude = [
    # Colonnes originales de santé mentale
    'Do you have Depression?',
    'Do you have Anxiety?',
    'Do you have Panic attack?',
    'Did you seek any specialist for a treatment?',
    
    # Versions encodées de ces colonnes (CRITIQUE!)
    'Do you have Depression?_encoded',
    'Do you have Anxiety?_encoded',
    'Do you have Panic attack?_encoded',
    'Did you seek any specialist for a treatment?_encoded',
    
    # Variables cibles
    'mental_health_issue',
    'mental_health_score',
    
    # Autres colonnes non pertinentes
    'Timestamp',
    'Cluster'  # Résultat du clustering, pas une feature
]

print("Colonnes à exclure:")
for col in columns_to_exclude:
    if col in df_clean.columns:
        print(f"  ✓ {col}")

# =============================================================================
# ÉTAPE 3: CRÉER LES FEATURES PROPRES
# =============================================================================
print("\n--- Création des features sans data leakage ---\n")

# Features autorisées (uniquement les caractéristiques démographiques et académiques)
valid_features = []

# 1. Features numériques directes
if 'Age' in df_clean.columns:
    valid_features.append('Age')

if 'What is your CGPA?' in df_clean.columns:
    # Convertir CGPA en numérique si nécessaire
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

# 2. Features encodées AUTORISÉES (pas liées à la santé mentale)
safe_encoded_columns = [
    'Choose your gender_encoded',
    'Your current year of Study_encoded',
    'What is your course?_encoded',
    'Marital status_encoded'
]

for col in safe_encoded_columns:
    if col in df_clean.columns:
        valid_features.append(col)

print(f"Features valides sélectionnées: {len(valid_features)}")
for feat in valid_features:
    print(f"  ✓ {feat}")

# Vérification qu'on a bien des features
if len(valid_features) == 0:
    print("\n⚠️ ERREUR: Aucune feature valide trouvée!")
    print("Colonnes disponibles dans le dataset:")
    print(df_clean.columns.tolist())
else:
    print(f"\n✓ {len(valid_features)} features valides identifiées")

# =============================================================================
# ÉTAPE 4: CRÉER LES MATRICES X ET Y PROPRES
# =============================================================================
print("\n--- Création des matrices X et y ---\n")

# Créer X sans data leakage
X_clean = df_clean[valid_features].copy()

# Gérer les valeurs manquantes
X_clean = X_clean.fillna(X_clean.median())

print(f"X_clean shape: {X_clean.shape}")
print(f"Valeurs manquantes: {X_clean.isnull().sum().sum()}")

# Créer les variables cibles
y_classification = df_clean['mental_health_issue'].copy()
y_regression = df_clean['mental_health_score'].copy()

print(f"y_classification shape: {y_classification.shape}")
print(f"y_regression shape: {y_regression.shape}")

# =============================================================================
# ÉTAPE 5: NORMALISATION CORRECTE
# =============================================================================
print("\n--- Normalisation des données ---\n")

# Important: normaliser APRÈS avoir créé X_clean
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns)

print(f"✓ Données normalisées: {X_scaled.shape}")

# =============================================================================
# ÉTAPE 6: VÉRIFICATION DE DATA LEAKAGE
# =============================================================================
print("\n--- Vérification finale de data leakage ---\n")

# Calculer les corrélations avec la cible
correlations = []
for col in X_clean.columns:
    corr = X_clean[col].corr(y_classification)
    correlations.append({
        'Feature': col,
        'Correlation': abs(corr)
    })

corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)

print("Corrélations avec la cible (mental_health_issue):")
print(corr_df.to_string(index=False))

# Vérifier s'il y a des corrélations suspectes
suspicious = corr_df[corr_df['Correlation'] > 0.95]
if len(suspicious) > 0:
    print("\n⚠️ ATTENTION: Features avec corrélation > 0.95 détectées:")
    print(suspicious)
else:
    print("\n✓ Aucune corrélation suspecte détectée (toutes < 0.95)")

# =============================================================================
# ÉTAPE 7: TEST RAPIDE DE PERFORMANCE RÉALISTE
# =============================================================================
print("\n--- Test rapide avec Random Forest ---\n")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)

# Modèle avec paramètres conservateurs
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,  # Limité pour éviter overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Validation croisée
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Entraînement
rf.fit(X_train, y_train)

# Évaluation
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

print(f"\nRésultats Random Forest:")
print(f"  Accuracy Train: {acc_train:.4f}")
print(f"  Accuracy Test:  {acc_test:.4f}")
print(f"  F1-Score Test:  {f1_test:.4f}")
print(f"  Écart Train-Test: {acc_train - acc_test:.4f}")

# Interprétation
print("\n--- Interprétation des résultats ---")
if acc_test >= 0.99:
    print("⚠️ PROBLÈME PERSISTANT: Accuracy trop élevée (>= 0.99)")
    print("   Il reste probablement du data leakage")
    print("   Vérifiez manuellement les colonnes utilisées")
elif acc_test >= 0.90:
    print("✓ BON: Accuracy réaliste (0.90-0.99)")
    print("   Pas de data leakage évident")
elif acc_test >= 0.75:
    print("✓ EXCELLENT: Accuracy réaliste et robuste (0.75-0.90)")
    print("   Modèle bien généralisé, pas de data leakage")
else:
    print("⚠️ ATTENTION: Accuracy faible (< 0.75)")
    print("   Vérifiez que vous avez assez de features pertinentes")

if acc_train - acc_test > 0.10:
    print("\n⚠️ OVERFITTING DÉTECTÉ:")
    print(f"   Écart train-test = {acc_train - acc_test:.4f} > 0.10")
    print("   → Augmenter la régularisation ou réduire la complexité")
else:
    print("\n✓ PAS D'OVERFITTING:")
    print(f"   Écart train-test = {acc_train - acc_test:.4f} <= 0.10")

# =============================================================================
# ÉTAPE 8: SAUVEGARDE DES DONNÉES CORRIGÉES
# =============================================================================
print("\n--- Sauvegarde des données corrigées ---\n")

X_scaled.to_csv('data/X_scaled_corrected.csv', index=False)
y_classification.to_csv('data/y_classification_corrected.csv', index=False)
y_regression.to_csv('data/y_regression_corrected.csv', index=False)

import joblib
joblib.dump(scaler, 'models/scaler_corrected.pkl')

print("✓ Données sauvegardées:")
print("  - data/X_scaled_corrected.csv")
print("  - data/y_classification_corrected.csv")
print("  - data/y_regression_corrected.csv")
print("  - models/scaler_corrected.pkl")

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print("\n" + "=" * 80)
print("RÉSUMÉ DE LA CORRECTION")
print("=" * 80)
print(f"\n✓ Features originales exclues: {len(columns_to_exclude)}")
print(f"✓ Features valides conservées: {len(valid_features)}")
print(f"✓ Échantillons: {len(X_scaled)}")
print(f"✓ Test accuracy: {acc_test:.4f}")
print(f"✓ Écart train-test: {acc_train - acc_test:.4f}")

if acc_test < 0.99 and acc_train - acc_test <= 0.10:
    print("\n🎉 SUCCÈS: Data leakage corrigé!")
    print("   Vous pouvez maintenant utiliser ces données pour votre modélisation")
else:
    print("\n⚠️ Il peut rester des problèmes - Vérifications supplémentaires nécessaires")

print("\n" + "=" * 80)
print("PROCHAINES ÉTAPES")
print("=" * 80)
print("""
1. Utilisez X_scaled_corrected.csv au lieu de X_scaled.csv
2. Relancez tous vos modèles avec ces nouvelles données
3. Les performances devraient être réalistes (0.75-0.90)
4. Si accuracy > 0.95, vérifiez encore les features
""")
