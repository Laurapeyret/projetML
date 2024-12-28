import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
import joblib
import warnings

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore')

# Définir le chemin du fichier
file_path = 'data/flights.csv'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Le fichier spécifié n'existe pas: {file_path}")

# Charger les données
df = pd.read_csv(file_path)

# Aperçu des données
print(df.head())
print(df.info())
print(df.describe())

# Prendre un échantillon de données pour accélerer l'entraînement
df = df.sample(n=50000, random_state=42)

# Identifiez les valeurs manquantes
print("Valeurs manquantes avant traitement :")
print(df.isnull().sum())

# Préparation des données
def preprocess_data(df):
    # Supprimer les colonnes non pertinentes
    df = df.drop(columns=['YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'CANCELLATION_REASON', 
                          'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 
                          'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'])
    
    # Imputations des valeurs manquantes avec la moyenne pour les colonnes numériques
    imputer = SimpleImputer(strategy='mean')
    numeric_features = ['DEPARTURE_DELAY', 'TAXI_OUT', 'TAXI_IN', 'ARRIVAL_DELAY']
    df[numeric_features] = imputer.fit_transform(df[numeric_features])
    
    # Suppression des lignes restantes avec des valeurs manquantes pour simplifier
    df = df.dropna()

    # Transformation des variables catégoriques
    df = pd.get_dummies(df, columns=['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
    
    # Standardisation des caractéristiques
    scaler = StandardScaler()
    df[['DEPARTURE_DELAY', 'TAXI_OUT', 'TAXI_IN']] = scaler.fit_transform(df[['DEPARTURE_DELAY', 'TAXI_OUT', 'TAXI_IN']])
    
    # Création de nouvelles caractéristiques
    df['NEW_FEATURE'] = df['MONTH'] * df['DAY_OF_WEEK']
    
    return df

df = preprocess_data(df)

# Assurez-vous qu'il n'y a plus de NaN
print("Valeurs manquantes après traitement :")
print(df.isnull().sum())

# Visualisation des retards
plt.figure(figsize=(10, 6))
sns.histplot(df['ARRIVAL_DELAY'], bins=50, kde=True)
plt.title('Distribution des retards')
plt.xlabel('Retard (minutes)')
plt.ylabel('Fréquence')
plt.savefig('retards_distribution.png')

# Division des données en ensembles d'entraînement et de test
X = df.drop('ARRIVAL_DELAY', axis=1)
y = df['ARRIVAL_DELAY'].apply(lambda x: 1 if x > 15 else 0)  # Retard binaire
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraînement et évaluation des modèles
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Decision Tree': DecisionTreeClassifier()
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'f1': f1,
            'roc_auc': roc_auc
        }
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred))

        # Sauvegarde du modèle
        joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.pkl')
    
    return results

# Fonction principale
def main():
    # Visualiser la heatmap des corrélations après avoir supprimé les colonnes non numériques
    numeric_cols = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
    plt.title('Heatmap of Correlations')
    plt.savefig('correlations_heatmap.png')
    
    results = train_and_evaluate_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()

