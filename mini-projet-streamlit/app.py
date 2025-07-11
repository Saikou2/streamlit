import streamlit as st 
from pathlib import Path
import pandas as pd 
import os
from plotly import express as px

current_dir=Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file=current_dir/"style.css"

with open(css_file) as f : 
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

fl = st.file_uploader(":file_folder: Upload a file", type=["csv", "xls","xlsx"])

if fl is not None:
    st.write(fl.name)
    df = pd.read_csv(fl)
else:
    # os.chdir(r'C:/M1 DSIA/streamlite')
    df = pd.read_csv('donnees_ventes_etudiants_mis_a_jour.csv')

st.title('Tableau de bord pour les ventes')

st.sidebar.header('Choisi ton filter')

col1, col2 = st.columns(2)

df['order_date'] = pd.to_datetime(df["order_date"])

startdate = df["order_date"].min()
enddate = df["order_date"].max()
df_filtered = df.copy()

region = st.sidebar.multiselect("Region", df['Region'].unique())
if region:
    df_filtered = df_filtered[df_filtered['Region'].isin(region)]

state = st.sidebar.multiselect("State", df_filtered['state_full'].unique())
if state:
    df_filtered = df_filtered[df_filtered['state_full'].isin(state)]

county = st.sidebar.multiselect("County", df_filtered['County'].unique())
if county:
    df_filtered = df_filtered[df_filtered['County'].isin(county)]

city = st.sidebar.multiselect("City", df_filtered['City'].unique())
if city:
    df_filtered = df_filtered[df_filtered['City'].isin(city)]

status_filter = st.sidebar.multiselect("Sélectionnez le statut de la commande", df['status'].unique())
if status_filter:
    df_filtered = df_filtered[df_filtered['status'].isin(status_filter)]

col1, col2 = st.columns(2)

with col1:
    date1 = st.date_input("Date debut", startdate, min_value=startdate, max_value=enddate)

with col2:
    date2 = st.date_input("Date fin", enddate, min_value=startdate, max_value=enddate)

date1 = pd.to_datetime(date1)
date2 = pd.to_datetime(date2)

df_filtered = df_filtered[(df_filtered['order_date'] >= date1) & (df_filtered['order_date'] <= date2)]

sales_by_category = df_filtered.groupby('category')['total'].sum().reset_index()
fig_bar = px.bar(sales_by_category, x='category', y='total', color='category',
                 labels={'total': 'Nombre total de ventes', 'category': 'Catégorie'},
                 title="Nombre total de ventes par Catégorie")

sales_by_region = df_filtered.groupby('Region')['total'].sum().reset_index()
fig_pie = px.pie(sales_by_region, values='total', names='Region', 
                 title="Pourcentage des ventes par Région", 
                 hole=0.3)  
fig_bar.update_traces(showlegend=False)
col1, col2 = st.columns(2)
col1.plotly_chart(fig_bar, use_container_width=True)
col2.plotly_chart(fig_pie, use_container_width=True)

# Affichage des KPI en utilisant st.metric
st.write(" Indicateurs clés de performance")

total_sales = df_filtered['total'].sum()
distinct_customers = df_filtered['cust_id'].nunique()
distinct_orders = df_filtered['order_id'].nunique()

st.markdown(f"""
    <div class="kpi-box">
        <div class="total">
            <p>Total des ventes</p>
            <p> <b> {total_sales} </b> </p>
        </div>
        <div class="client">
            <p>Clients uniques</p>
            <p> <b> {distinct_customers} </b> </p>
        </div>
        <div class="commande">
            <p>Total des commandes</p>
            <p> <b> {distinct_orders} </b></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Top 10 des meilleurs clients
top_clients = df_filtered.groupby('full_name')['total'].sum().nlargest(10).reset_index()
fig_top_clients = px.bar(top_clients, x='full_name', y='total',color='full_name', 
                         labels={'full_name': 'Nom du Client', 'Total': 'Total des Ventes'},
                         title="Top 10 des Meilleurs Clients")
fig_top_clients.update_layout(xaxis={'categoryorder': 'total descending'})

st.plotly_chart(fig_top_clients, use_container_width=True)

# Histogramme de la répartition de l'âge des clients
fig_age_hist = px.histogram(df_filtered, x='age', nbins=10, color="age",
                            labels={'Age': 'Âge des clients'},
                            title="Répartition de l'âge des clients")
fig_age_hist.update_layout(bargap=0.1) 


gender_counts = df_filtered['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']
gender_counts['Percentage'] = (gender_counts['Count'] / gender_counts['Count'].sum()) * 100

fig_gender_bar = px.bar(gender_counts, x='Gender', y='Count', text='Percentage', color="Count",
                        labels={'Gender': 'Genre', },
                        title="Répartition des genres")
fig_gender_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

# Affichage des graphiques sur une nouvelle ligne
st.write("### Répartition de l'âge et des genres")
col1, col2 = st.columns(2)
col1.plotly_chart(fig_age_hist, use_container_width=True)
col2.plotly_chart(fig_gender_bar, use_container_width=True)

st.write('Affichage des donnes de vente')
st.write(df_filtered)

# 7. times series

df['order_date'] = pd.to_datetime(df['order_date'])

# Créer une colonne 'Year_Month' pour représenter le mois et l'année
df['Year_Month'] = df['order_date'].dt.to_period('M').astype(str)

# Calculer le total des ventes par mois et année
sales_by_month = df.groupby('Year_Month')['total'].sum().reset_index()
sales_by_month['Year_Month'] = pd.to_datetime(sales_by_month['Year_Month'])

# Tracer la courbe des ventes par mois-année
fig_sales_month = px.line(sales_by_month, x='Year_Month', y='total', 
                          labels={'Year_Month': 'Mois-Année', 'Total': 'Total des Ventes'},
                          title="Nombre total de ventes par Mois")
fig_sales_month.update_layout(xaxis_title="Mois-Année", yaxis_title="Total des Ventes")

# Afficher la courbe dans Streamlit
st.write("### Évolution du nombre total de ventes par Mois")
st.plotly_chart(fig_sales_month, use_container_width=True)

state_sales = df.groupby(['state_full', 'Latitude', 'Longitude']).agg(total_sales=('total', 'sum')).reset_index()

# Création de la carte
fig = px.scatter_geo(
    state_sales,
    lat='Latitude',
    lon='Longitude',
    hover_name='state_full',
    hover_data={'total_sales': True, 'Latitude': False, 'Longitude': False},
    size='total_sales',
    size_max=30,
    scope='usa',
    color='total_sales',
    color_continuous_scale='Viridis',
    title="Nombre total de ventes par État",
    text='state_full'  
)

fig.update_traces(textposition='top center') 
st.plotly_chart(fig)

