import streamlit as st
from sklearn import datasets 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

## Création de l'en-tête de l'application
st.header('''
Explorez différents Classifieurs 
Quel est le meilleur ?
''')

## Mettre la sélection à gauche pour le dataset
dataset_name = st.sidebar.selectbox('Sélectionnez le Dataset', ('Iris', 'Cancer du sein', 'Wine'))
st.write(dataset_name)

## Sélectionner le classifieur
classifier_name = st.sidebar.selectbox('Sélectionnez le Classifieur', ('KNN', 'SVM', 'Random Forest'))

### Fonction pour sélectionner le dataset
def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Cancer du sein':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y    

X, y = get_dataset(dataset_name)

st.write('Aperçu des données :')
st.dataframe(X)

st.write('Dimensions du dataset :', X.shape)
st.write('Nombre de classes :', len(np.unique(y)))

### Trouver les meilleurs paramètres automatiquement

def find_best_params(clf_name, X_train, y_train):
    best_params = {}
    best_acc = 0

    if clf_name == 'KNN':
        best_k = 1
        for k in range(1, 16):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            accuracy = knn.score(X_train, y_train)
            if accuracy > best_acc:
                best_acc = accuracy
                best_k = k
        best_params['K'] = best_k
        st.write(f"Meilleur K trouvé: {best_k} avec une précision de {best_acc:.2f}")
    
    elif clf_name == 'SVM':
        best_c = 0.01
        for C in np.arange(0.01, 10.1, 0.5):
            svm = SVC(C=C)
            svm.fit(X_train, y_train)
            accuracy = svm.score(X_train, y_train)
            if accuracy > best_acc:
                best_acc = accuracy
                best_c = C
        best_params['C'] = best_c
        st.write(f"Meilleur C trouvé: {best_c:.2f} avec une précision de {best_acc:.2f}")
    
    elif clf_name == 'Random Forest':
        best_depth = 2
        best_estimators = 10
        for depth in range(2, 16):
            for n_estimators in range(10, 101, 1):
                rf = RandomForestClassifier(max_depth=depth, n_estimators=n_estimators)
                rf.fit(X_train, y_train)
                accuracy = rf.score(X_train, y_train)
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_depth = depth
                    best_estimators = n_estimators
        best_params['max_depth'] = best_depth
        best_params['n_estimators'] = best_estimators
        st.write(f"Meilleure profondeur trouvée: {best_depth} et meilleur n_estimators: {best_estimators} avec une précision de {best_acc:.2f}")
    
    return best_params

## Séparer les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

## Trouver les meilleurs paramètres pour le classifieur sélectionné
best_params = find_best_params(classifier_name, X_train, y_train)

### Ajouter l'interface pour ajuster les paramètres après avoir détecté les meilleurs
def add_parameter_ui(clf_name, best_params):
    params = {}
    
    if clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15, value=best_params['K'])  # Détection du meilleur K
        params['K'] = K
        
    elif clf_name == 'SVM':
        C = st.sidebar.slider("C", 0.01, 10.0, value=best_params['C'])  # Détection du meilleur C
        params['C'] = C
        
    else:
        max_depth = st.sidebar.slider('Profondeur Max', 2, 15, value=best_params['max_depth'])  # Meilleure profondeur détectée
        n_estimators = st.sidebar.slider('Nombre d\'estimations', 10, 100, value=best_params['n_estimators'])  # Meilleurs n_estimators détectés
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    
    return params

## Appeler l'interface utilisateur pour ajuster les paramètres
params = add_parameter_ui(classifier_name, best_params)

### Créer et entraîner le modèle avec les paramètres optimisés
def get_classifier(clf_name, params):
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'SVM':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
    return clf

clf = get_classifier(classifier_name, params)

## Entraînement et prédiction
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

## Calculer la précision
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifieur = {classifier_name}')
st.write(f'Précision = {acc}')

## Visualisation avec PCA
pca = PCA(2)
x_projete = pca.fit_transform(X_test)

st.write('La variance expliquée par PCA :', pca.explained_variance_ratio_)

x1 = x_projete[:, 0]
x2 = x_projete[:, 1]

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(x1, x2, c=y_test, alpha=0.8, cmap='viridis')
ax1.set_title('Labels réels')

ax2.scatter(x1, x2, c=y_pred, alpha=0.8, cmap='viridis')
ax2.set_title('Labels prédits')

st.pyplot(fig)
