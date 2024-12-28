# Projet Machine Learning 


1.  Nom du dataset utilisé : Flight Delay Prediction

Ce projet utilise plusieurs modèles de machine learning pour prédire les retards des vols aux États-Unis.



2.  Objectif du Projet

Le but de ce projet est de prédire le retard des vols en utilisant des modèles de machine learning classiques. Les données utilisées sont issues du dataset "US Flight Delay" qui est disponible sur Kaggle à l'url suivant : https://www.kaggle.com/datasets/usdot/flight-delays. Il faut récupérer les données contenues dans le fichier flights.csv el les mettre dans le dossier data non fourni par le GitHub. 



3.  Structure du Projet

Developer/ projetML/ 	├── data/ 

				└── flights.csv 

			├── flight_delay_prediction.py 

			├── README.md

			└── requirements.txt


Autrement dit, 

- Le dossier data contient le fichier CSV des données de vols (flights.csv)
- Le fichier flight_delay_prediction.py est le script Python principal qui est chargé de charger les données, les nettoyer, entraîner les modèles et évaluer leurs performances.
- Le fichier README.md qui est celui ci et qui permet la documentation du projet.
- Le fichier requirements.txt qui est liste des dépendances nécessaires.



4. Installation

	a) Cloner le dépôt GitHub

Pour obtenir le code source de ce projet, on doit cloner le dépôt GitHub. Cela permet de créer une copie locale de ce projet sur notre ordinateur. Pour ce faire, il faut utiliser la commande suivante : 

git clone https://github.com/Laurapeyret/flight_delay_prediction.git
cd flight_delay_prediction

	b) Installer les dépendances

Le fichier requirements.txt liste toutes les bibliothèques Python nécessaires pour ce projet. Pour installer ces bibliothèques, il faut utiliser pip avec la commande suivante :

pip install -r requirements.txt



5. Exécution

Le script principal flight_delay_prediction.py charge les données, les nettoie, entraîne plusieurs modèles de machine learning, et évalue leurs performances. Pour exécuter ce script, on utilise la commande suivante :

python3 flight_delay_prediction.py



6. Détails du Script

	a) Chargement des données

Le script commence par charger les données depuis le fichier CSV fourni dans le répertoire data. Les données contiennent des informations sur les vols, y compris les horaires, les aéroports de départ et d'arrivée, et les retards.


	b) Pré-traitement des données

Le script continue avec un pré-traitement des données, incluant :

- Suppression des colonnes non pertinentes avec l'élimination des colonnes qui ne sont pas nécessaires pour l'analyse.
- Gestion des valeurs manquantes avec l'utilisation d'un imputer pour remplacer les valeurs manquantes avec la moyenne des colonnes.
- Transformation des variables catégoriques avec la conversion des variables catégoriques (comme les codes des compagnies aériennes et des aéroports) en variables numériques à l'aide du one-hot encoding.
- Standardisation des caractéristiques avec la mise à l'échelle des caractéristiques numériques pour qu'elles aient une moyenne de zéro et un écart-type d'un.


	c) Division des données

Les données pré-traitées sont divisées en ensembles d'entraînement et de test (voici la partie du code correspondante) : 

		X = df.drop('ARRIVAL_DELAY', axis=1)
		y = df['ARRIVAL_DELAY'].apply(lambda x: 1 if x > 15 else 0)  # Binariser la variable cible
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


	d) Entraînement des modèles

Le script entraîne plusieurs modèles de machine learning sur les données d'entraînement :

- Régression logistique
- Forêt aléatoire (Random Forest)
- Boosting de gradient
- Arbre de décision
Pour chaque modèle, le script calcule des métriques de performance et sauvegarde le modèle :

	models = {
	    'Logistic Regression': LogisticRegression(max_iter=1000),
	    'Random Forest': RandomForestClassifier(),
	    'Gradient Boosting': GradientBoostingClassifier(),
	    'Decision Tree': DecisionTreeClassifier()
	}

	for name, model in models.items():
	    model.fit(X_train, y_train)
	    y_pred = model.predict(X_test)
	    accuracy = accuracy_score(y_test, y_pred)
	    f1 = f1_score(y_test, y_pred)
	    roc_auc = roc_auc_score(y_test, y_pred)
	    print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
	    print(classification_report(y_test, y_pred))
	    joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.pkl')


	e) Visualisation des résultats

Le script génère également quelques visualisations pour analyser les données et les résultats, telles que des heatmaps de corrélation et des histogrammes des retards.


	f) Résultats

Les performances des modèles sont affichées dans le terminal et les meilleurs modèles sont sauvegardés dans des fichiers .pkl. Voici quelques exemples de résultats obtenus :

Exemple de Résultats

Logistic Regression :
	Accuracy: 95.69%
	F1-Score: 87.00%
	ROC AUC: 90.79%

Random Forest :
	Accuracy: 94.47%
	F1-Score: 82.30%
	ROC AUC: 86.49%

Gradient Boosting :
	Accuracy: 95.43%
	F1-Score: 86.24%
	ROC AUC: 90.39%

Decision Tree :
	Accuracy: 93.95%
	F1-Score: 82.46%
	ROC AUC: 89.30%




7. Sauvegarde des Modèles

Les modèles entraînés sont sauvegardés dans des fichiers .pkl pour une utilisation future :


├── logistic_regression_model.pkl
├── random_forest_model.pkl
├── gradient_boosting_model.pkl
└── decision_tree_model.pkl




8. Données

	a) Source des Données

Les données utilisées dans ce projet proviennent du dataset "US Flight Delay" disponible sur Kaggle. Le fichier flights.csv contient les informations des vols pour l'année 2015.


	b) Traitement des Données

Le script de traitement des données effectue les étapes suivantes :

	1.Chargement des données depuis flights.csv.
	2.Nettoyage des données : suppression des colonnes non pertinentes et traitement des valeurs manquantes.
	3.Transformation des variables catégoriques en variables numériques.
	4.Standardisation des caractéristiques.
	5.Création de nouvelles caractéristiques.







