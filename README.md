# Projet Machine Learning - Student Mental Health

## 📋 Description
Ce projet applique diverses techniques de Machine Learning sur le dataset Student Mental Health 
pour analyser et prédire les problèmes de santé mentale chez les étudiants.

## 📂 Structure du Projet

```
.
├── data/                          # Données brutes et traitées
│   ├── Student Mental health.csv
│   ├── student_mental_health_cleaned.csv
│   ├── X_scaled.csv
│   ├── y_regression.csv
│   └── y_classification.csv
├── models/                        # Modèles entraînés
│   ├── regression/
│   ├── classification/
│   ├── clustering/
│   ├── dimensionality_reduction/
│   └── recommendation/
├── visualizations/                # Graphiques et visualisations
├── results/                       # Résultats et rapports
│   ├── final_summary.csv
│   ├── final_report.html
│   └── *_results.json
└── notebooks/                     # Notebooks Jupyter

```

## 🎯 Objectifs Pédagogiques Validés

✅ AA1: Concepts clés du Machine Learning
✅ AA2: Phases d'un projet ML
✅ AA3: Préparation des données avec Python
✅ AA4: Modèles de régression
✅ AA5: Méthodes de classification
✅ AA6: Modèles de segmentation

## 🔧 Technologies Utilisées

- Python 3.x
- NumPy, Pandas
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

## 📊 Dataset

- **Source** : [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Taille** : 303 patients, 14 attributs
- **Objectif** : Prédiction de maladie cardiaque


## 📊 Modèles Implémentés

### Régression
- Régression Linéaire Simple et Multiple
- Régression Polynomiale (degrés 1-4)

### Classification
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Arbre de Décision
- Random Forest
- XGBoost

### Clustering
- K-Means (k optimal: 3)

### Réduction Dimensionnelle
- PCA (9 composantes pour 95% variance)

### Systèmes de Recommandation
- Content-Based Filtering
- Collaborative Filtering
- Hybrid Recommender System

## 📈 Résultats Clés

     Catégorie         Méthode Métrique principale
    Régression   Linear Simple          R²=-0.0086
    Régression Linear Multiple           R²=1.0000
Classification             KNN          Acc=0.9524
Classification    SVM (linear)          Acc=1.0000
Classification   Decision Tree          Acc=1.0000
Classification   Random Forest          Acc=1.0000
Classification         XGBoost          Acc=1.0000
    Clustering   K-Means (k=3)   Silhouette=0.2158
Réduction dim.   PCA (9 comp.)      Variance=96.7%
## 🚀 Utilisation

1. Télécharger le dataset depuis Kaggle
2. Placer le fichier CSV dans le dossier `data/`
3. Exécuter le script principal: `python ml_complete_project.py`
4. Consulter les résultats dans `results/final_report.html`

## 📝 Auteur

Projet académique - Module SI-17
Date: 2025-11-19

## 📄 Licence

Projet éducatif - Usage académique uniquement
