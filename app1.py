import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Charger et préparer les données
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("Invistico_Airline.csv")  
    except FileNotFoundError:
        st.error("Le fichier 'Invistico_Airline.csv' n'a pas été trouvé. Veuillez vérifier le chemin.")
        return None, None, None, None, None, None, None, None, None
    
    df.columns = df.columns.str.strip()
    if 'satisfaction' in df.columns:
        df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

    X = df.drop(columns=['satisfaction'])  
    y = df['satisfaction']
    
    # Prétraitement des données
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

    num_pipeline = Pipeline([ 
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([ 
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    full_preprocess = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    X_processed = full_preprocess.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, train_size=0.8, random_state=42)
    
    return df, full_preprocess, X_train, X_test, y_train, y_test, X.columns

# Charger les modèles
@st.cache_resource
def load_models(X_train, y_train):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    
    return knn, model_nb

# Interface utilisateur Streamlit
def app():
    st.title("Prédiction de la Satisfaction Client")
    st.write("Cette application utilise des modèles de machine learning pour prédire la satisfaction client.")
    
    df, full_preprocess, X_train, X_test, y_train, y_test, X_columns = load_and_preprocess_data()
    if df is None:
        return
    
    knn, model_nb = load_models(X_train, y_train)

    # Section d'évaluation du modèle
    st.header("Évaluation des Modèles")
    model_eval = st.selectbox("Choisissez un modèle pour l'évaluation", ("KNN", "Naïve Bayes"))

    if st.button("Évaluer le Modèle"):
        y_test = y_test.astype(int)

        if model_eval == "KNN":
            y_test_pred = knn.predict(X_test)
            st.write("Accuracy:", accuracy_score(y_test, y_test_pred))
            st.write("Classification Report:\n", classification_report(y_test, y_test_pred))
        elif model_eval == "Naïve Bayes":
            y_test_pred = model_nb.predict(X_test)
            st.write("Accuracy:", accuracy_score(y_test, y_test_pred))
            st.write("Classification Report:\n", classification_report(y_test, y_test_pred))

    # Section de prédiction avec de nouvelles données
    st.header("Prédiction avec de Nouvelles Données")
    st.write("Veuillez saisir de nouvelles données pour effectuer une prédiction.")

    user_input = {}
    for col in X_columns:
        if df[col].dtype in ['float64', 'int64']:
            user_input[col] = st.number_input(f"{col} (numérique)", value=0, key=col)
        else:
            options = df[col].unique()
            user_input[col] = st.selectbox(f"{col} (catégoriel)", options, key=col)

    model_choice = st.selectbox("Choisissez un modèle pour la prédiction", ("KNN", "Naïve Bayes"))

    if st.button("Prédire"):
        input_data = pd.DataFrame([user_input])

        processed_input_data = full_preprocess.transform(input_data)

        if model_choice == "KNN":
            probas = knn.predict_proba(processed_input_data)
            prediction = knn.predict(processed_input_data)
            satisfaction_prediction = "satisfait" if prediction[0] == 1 else "non satisfait"
            prob_satisfaction = probas[0][1]
        elif model_choice == "Naïve Bayes":
            probas = model_nb.predict_proba(processed_input_data)
            prediction = model_nb.predict(processed_input_data)
            satisfaction_prediction = "satisfait" if prediction[0] == 1 else "non satisfait"
            prob_satisfaction = probas[0][1]

        st.subheader("Résultat de la Prédiction")
        st.write(f"Le modèle prédit que le client est : *{satisfaction_prediction}*")
        st.write(f"Probabilité de satisfaction : *{prob_satisfaction:.2f}*")

if __name__ == "__main__":
    app()
