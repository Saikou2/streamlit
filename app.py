import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("Invistico_Airline.csv")
    return df

data = load_data()
st.title("Analyse de la Satisfaction Client")
st.write("Analyse Exploratoire des Données (EDA)")
st.write("Aperçu des Données", data.head())
st.write("Taille du dataset:", data.shape)

# Afficher des informations sur les colonnes
buffer = pd.DataFrame({
    "Colonnes": data.columns,
    "Type": data.dtypes,
    "Valeurs manquantes": data.isnull().sum()
}).astype({'Valeurs manquantes': int})
st.write("Informations sur les colonnes:", buffer)

# Séparation des caractéristiques numériques et catégorielles
num_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_features = data.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

# Suppression de la colonne cible des caractéristiques
if 'satisfaction' in num_features:
    num_features.remove('satisfaction')
if 'satisfaction' in cat_features:
    cat_features.remove('satisfaction')

# Préparation des données
X = data.drop(columns=['satisfaction'])
y = data['satisfaction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prétraitement
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first'))
])
full_preprocess = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

# Application du prétraitement aux données d'entraînement et de test
X_train = full_preprocess.fit_transform(X_train)
X_test = full_preprocess.transform(X_test)

# Réduction de dimension avec PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)
st.write("Variance expliquée par les composants PCA:", pca.explained_variance_ratio_)

# Visualisation PCA
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2']).assign(satisfaction=y_train.reset_index(drop=True))
fig, ax = plt.subplots()
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='satisfaction', ax=ax)
st.pyplot(fig)

# Choix et entraînement des modèles
st.write("Entraînement du Modèle")
model_choice = st.selectbox("Choisissez un modèle:", ["KNN", "Naive Bayes"])

if model_choice == "KNN":
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': range(1, 10)}
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
elif model_choice == "Naive Bayes":
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    model = model_nb

# Prédictions et évaluation
y_pred = model.predict(X_test)
st.write("Précision:", accuracy_score(y_test, y_pred))
st.text("Rapport de Classification:\n" + classification_report(y_test, y_pred))

# Visualisations supplémentaires
st.subheader("Carte de Chaleur des Corrélations")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data[num_features].corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.subheader("Distribution des Caractéristiques Numériques")
for feature in num_features:
    fig, ax = plt.subplots()
    sns.histplot(data[feature], kde=True, ax=ax)
    ax.set_title(f"Distribution de {feature}")
    st.pyplot(fig)

st.subheader("Distribution des Caractéristiques Catégorielles")
for feature in cat_features:
    fig, ax = plt.subplots()
    sns.countplot(x=feature, data=data, ax=ax)
    ax.set_title(f"Répartition de {feature}")
    st.pyplot(fig)

# Prédictions sur de nouvelles données
st.subheader("Prédictions avec de Nouvelles Données")
user_input = {}
for feature in num_features + cat_features:
    if feature in num_features:
        user_input[feature] = st.number_input(f"{feature} (numérique)", value=0)
    else:
        user_input[feature] = st.selectbox(f"{feature} (catégoriel)", data[feature].unique())

if st.button("Prédire"):
    input_data = pd.DataFrame([user_input])
    processed_input_data = full_preprocess.transform(input_data)

    if model_choice == "KNN":
        probas = model.predict_proba(processed_input_data)
        satisfaction_prediction = "Satisfait" if model.predict(processed_input_data)[0] == 1 else "Non Satisfait"
        prob_satisfaction = probas[0][1]
    elif model_choice == "Naive Bayes":
        probas = model.predict_proba(processed_input_data)
        satisfaction_prediction = "Satisfait" if model.predict(processed_input_data)[0] == 1 else "Non Satisfait"
        prob_satisfaction = probas[0][1]

    st.subheader("Résultat de la Prédiction")
    st.write(f"Le modèle prédit que le client est : *{satisfaction_prediction}*")
    st.write(f"Probabilité de satisfaction : *{prob_satisfaction:.2f}*")
