# Projet IA – Machine Learning et Mise en Production Simulée

## Présentation du projet
Ce projet a pour objectif de mettre en pratique l’ensemble des étapes d’un projet de Machine Learning,
depuis l’analyse et la préparation des données jusqu’à la mise à disposition du modèle via une API,
dans un contexte simulant une mise en production.

L’objectif n’est pas d’obtenir le modèle le plus performant possible, mais de comprendre le
fonctionnement global d’un projet IA de bout en bout.

---

## Jeu de données
Le projet utilise le **California Housing Dataset**, un jeu de données réel couramment utilisé
pour des problèmes de régression.

La variable cible est :
- `MedHouseVal` : valeur médiane des maisons

Les variables explicatives décrivent des caractéristiques socio-économiques et géographiques
(revenus, population, latitude, longitude, etc.).

---

## Étapes du projet

### 1. Exploration et préparation des données
- Chargement et exploration du jeu de données
- Vérification des types de données
- Analyse statistique descriptive
- Suppression des doublons et des valeurs manquantes
- Séparation des features (X) et de la variable cible (y)

---

### 2. Entraînement et comparaison des modèles
Plusieurs modèles de régression ont été entraînés et comparés :
- Régression linéaire
- Régression Ridge
- K-Nearest Neighbors (KNN)
- Random Forest

Les performances ont été évaluées à l’aide des métriques :
- RMSE (Root Mean Squared Error)
- R² (coefficient de détermination)

Le modèle **RandomForest** a présenté les meilleures performances et a été sélectionné comme
modèle final.

---

### 3. Sauvegarde du modèle
Le modèle sélectionné a été sauvegardé à l’aide de `joblib` afin de pouvoir être réutilisé
sans réentraîner le modèle.

---

### 4. Création d’une API (Agent IA)
Le modèle de Machine Learning a été intégré dans une API développée avec **FastAPI**.

Fonctionnalités de l’API :
- Endpoint `/predict` permettant d’obtenir une prédiction à partir de nouvelles données
- Validation des entrées avec Pydantic
- Endpoint `/health` pour vérifier l’état du service

L’API agit comme un **agent IA** capable de produire des prédictions automatiquement.

---

### 5. Tests unitaires
Des tests unitaires ont été mis en place avec **pytest** afin de vérifier :
- Le bon fonctionnement de l’API
- La disponibilité des endpoints
- La validité des réponses retournées

Ces tests permettent de garantir la qualité du code avant toute mise en production.

---

### 6. Pipeline CI/CD et simulation de mise en production
Une pipeline CI/CD a été mise en place avec **GitHub Actions**.

Fonctionnement :
- Le développement se fait sur une branche `dev`
- Les tests sont validés sur une branche `test`
- La branche `main` représente la production

La pipeline :
- Installe automatiquement les dépendances
- Vérifie les imports critiques
- Exécute les tests unitaires
- Simule une validation avant mise en production

Cette organisation permet de simuler un environnement professionnel de déploiement.

---

## Technologies utilisées
- Python
- Pandas
- Scikit-learn
- FastAPI
- Pytest
- GitHub Actions
- Joblib

---

## Conclusion
Ce projet a permis de comprendre concrètement comment un modèle de Machine Learning peut être
développé, évalué, intégré dans une application et validé dans un contexte proche de la production.
