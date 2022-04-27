# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import matplotlib
from streamlit_option_menu import option_menu

#st.cache(persist=True)
st.set_page_config(layout="wide")
with st.sidebar:
    selected = option_menu("Menu", ["Contexte", 'Analyse', 'Graphs', 'Dataframe', 'Moyenne', 'Type'], 
            icons=['house', 'activity', 'graph-up', 'table', 'calculator', 'bar-chart-steps'], default_index=0)

total= pd.read_csv("limited.csv")

if selected == "Analyse":
    total.rename(columns={'CalYear_x': 'Annee'}, inplace=True)
    st.write("Temps moyen par mois par type d'alarme et par district")
    clist = total['IncGeo_BoroughName'].unique()
    clist.sort()
    alist = total['SpecialServiceType'].unique()
    alist.sort()
    district3 = st.multiselect("Select a district : ", clist, default="WESTMINSTER")
    intervention = st.multiselect("Select a intervention",alist, default="AFA")
    query = f"IncGeo_BoroughName=={district3} & SpecialServiceType=={intervention}"
    df_filtered = total.query(query)
    col1, col2 = st.columns(2)
    col1.metric("Nb intervention : ",len(df_filtered))
    if len(df_filtered)>0:
        fig = sns.relplot(x='month', y='AttendanceTimeSeconds',ci=None, height=8, aspect=3, kind='line', hue='Annee', data=df_filtered)
        plt.xlabel("Mois")
        plt.ylabel("Temps d'attente moyen (s)")
        st.pyplot(fig)
        col2.metric("Moyenne : ", int(df_filtered['AttendanceTimeSeconds'].mean()))
    else:
        st.write("Pas d'intervention de ce type")

elif selected == "Graphs":

    st.title("Les graphs :")
    option = st.selectbox('Quel graph ?', ('Nb_intervention_incident', 'Delai_incident', 'Nb_intervention_district','Delai_district','Delai_mois_annee', 'Nb_intervention_brigade', 'Nb_intervention_brigade_district', 'Nb_intervention_biens', 'Delai_biens', 'Intervention_date'))
    if option == "Nb_intervention_incident":
        with st.spinner("Génération du graph ...."):
            fig=px.histogram(total, x='SpecialServiceType', text_auto='.2s', height=600).update_xaxes(categoryorder='total descending')
            fig.update_layout(
                title="Nombre d'intervention par type d'incident",
                yaxis=dict(title="Nombre d'interventions"),
                xaxis=dict(title="Type d'intervention")
                )
            plt.xticks(rotation=90)
            st.plotly_chart(fig, use_container_width=True)

            labels = ['AFA','Fausse alarme','Feux', 'Autres', 'Inondation', 'Collision', 'Extraction']
            Afa=len(total[total['SpecialServiceType'] == 'AFA'])
            entryexit=len(total[total['SpecialServiceType'] == 'Effecting entry/exit'])
            Flooding=len(total[total['SpecialServiceType'] == 'Flooding'])
            Rtc=len(total[total['SpecialServiceType'] == 'RTC'])
            Fal=len(total[total['SpecialServiceType'] == 'False alarm - Good intent'])+len(total[total['SpecialServiceType'] == 'False alarm - Malicious'])
            Fire=len(total[total['SpecialServiceType'] == 'Primary Fire'])+len(total[total['SpecialServiceType'] == 'Secondary Fire'])
            others=len(total)-Fal-Afa-entryexit-Flooding-Rtc-Fire        
            values = [Afa, Fal, Fire, others, Flooding, Rtc, entryexit]
            fig=px.pie(names=labels, values=values)
            fig.update_traces(textfont_size=17)
            st.plotly_chart(fig, use_container_width=True)
            st.write("Road Traffic Collision (RTC)")
            st.write("Automatic Fire Alarms (AFA)")

    if option == "Nb_intervention_biens":
        st.write("Nombre de type de biens : ", len(total['PropertyType'].unique()))
        with st.spinner("Génération du graph ...."):
            fig=px.histogram(total, x='PropertyType', text_auto='.2s', height=600).update_xaxes(categoryorder='total descending')
            fig.update_layout(
                title="Nombre d'intervention par type de biens",
                yaxis=dict(title="Nombre d'interventions"),
                xaxis=dict(title="Type de biens")
                )
            plt.xticks(rotation=90)
            st.plotly_chart(fig, use_container_width=True)

    if option == "Nb_intervention_brigade_district":
        clist = total['IncGeo_BoroughName'].unique()
        clist.sort()
        district1 = st.selectbox("Select a district: ",clist, index=32)
        query = f"IncGeo_BoroughName=='{district1}'"
        df_filtered = total.query(query)
        somme=len(df_filtered['DeployedFromStation_Name'])
        temp=df_filtered['DeployedFromStation_Name'].value_counts()[:5]
        sommetop5=sum(temp.tolist())
        other=somme-sommetop5
        temp.index.name='val'
        temp=temp.append(pd.Series([other], index=['Autres']))
        temp=temp.reset_index()
        st.write("Top 5 sur ",len(df_filtered['DeployedFromStation_Name'].unique()))
        fig=px.pie(temp, names='index', values=0)
        #fig.update_traces(textfont_size=15)
        st.plotly_chart(fig, use_container_width=True)
      
    if option == "Nb_intervention_brigade":      
        with st.spinner("Génération du graph ...."):
            st.write("Nombre de brigate : ", len(total['DeployedFromStation_Name'].unique()))
            fig=px.histogram(total, x="DeployedFromStation_Name",  text_auto='.2s', height=600).update_xaxes(categoryorder='total descending')
            fig.update_layout(
              title="Nombre d'intervention total par brigade",
              yaxis=dict(title="Nombre d'interventions"),
              xaxis=dict(title="Nom de la brigade")
              )
            plt.xticks(rotation=90)
            st.plotly_chart(fig, use_container_width=True)

    if option == "Nb_intervention_district":
        with st.spinner("Génération du graph ...."):
            fig=px.histogram(total, x='IncGeo_BoroughName', text_auto='.2s', height=600).update_xaxes(categoryorder='total descending')
            fig.update_layout(
                title="Nombre d'intervention par district",
                yaxis=dict(title="Nombre d'interventions"),
                xaxis=dict(title="District")
                )
            plt.xticks(rotation=90)
            st.plotly_chart(fig, use_container_width=True)

    elif option=="Delai_district":
        with st.spinner("Génération du graph ...."):
            fig=px.histogram(total, x='IncGeo_BoroughName', y='AttendanceTimeSeconds', histfunc='avg', height=600, text_auto='.0f').update_xaxes(categoryorder='total descending')
            fig.update_layout(
                title="Temps d'attente moyen par district",
                yaxis=dict(title="Temps d'attente (s)"),
                xaxis=dict(title="District")
                )
            plt.xticks(rotation=90)
            st.plotly_chart(fig, use_container_width=True)

    elif option=="Delai_incident":
        with st.spinner("Génération du graph .."):
            fig=px.histogram(total, x='SpecialServiceType', y='AttendanceTimeSeconds', histfunc='avg', height=600, text_auto='.0f').update_xaxes(categoryorder='total descending')
            fig.update_layout(
                title="Temps d'attente moyen par type d'incident",
                yaxis=dict(title="Temps d'attente (s)"),
                xaxis=dict(title="Type d'incident")
                )
            plt.xticks(rotation=90)
            st.plotly_chart(fig, use_container_width=True)

            fig=px.histogram(total, x='SpecialServiceType', text_auto='.2s', height=600).update_xaxes(categoryorder='total descending')
            fig.update_layout(
                title="Nombre d'intervention par type d'incident",
                yaxis=dict(title="Nombre d'interventions"),
                xaxis=dict(title="Type d'intervention")
                )
            plt.xticks(rotation=90)
            st.plotly_chart(fig, use_container_width=True)

    elif option=="Delai_biens":
        with st.spinner("Génération du graph ."):
            st.write("Nombre de type de biens : ", len(total['PropertyType'].unique()))
            fig=px.histogram(total, x='PropertyType', y='AttendanceTimeSeconds', histfunc='avg', height=600, text_auto='.0f').update_xaxes(categoryorder='total descending')
            fig.update_layout(
                title="Temps d'attente moyen par type de biens",
                yaxis=dict(title="Temps d'attente (s)"),
                xaxis=dict(title="Type de biens")
                )
            plt.xticks(rotation=90)
            st.plotly_chart(fig, use_container_width=True)

    elif option=="Delai_mois_annee":
        with st.spinner("Génération du graph ...."):
            total.rename(columns={'CalYear_x': 'Annee'}, inplace=True)
            st.write("Delai d'intervention par mois et annees")
            fig = plt.figure(figsize=(15,16))
            fig = sns.relplot(x='month', y='AttendanceTimeSeconds',ci=None, height=8, aspect=3, kind='line', hue = 'Annee', data=total)
            plt.xlabel("Mois")
            plt.ylabel("Temps d'attente moyen (s)")
            st.pyplot(fig)

    elif option=="Intervention_date":
        with st.spinner("Génération du graph .."):
            total.rename(columns={'CalYear_x': 'annee'}, inplace=True)
            fig=px.histogram(total, x='month', color='annee', barmode='group', height=600, text_auto='.0f')
            fig.update_layout(
            title="Nombre d'intervention par mois et annee",
                   yaxis=dict(title="Nombre d'interventions"),
                   xaxis=dict(title="Mois"),
                   bargap=0.2
                   )
            plt.xticks(rotation=90)
            st.plotly_chart(fig, use_container_width=True)
        
elif selected == "Dataframe":
    temp= pd.read_csv("1000line.csv")
    st.title("Les 1000 premieres lignes sur 587073")
    st.dataframe(temp.head(1000))

elif selected == "Moyenne":
    total.rename(columns={'CalYear_x': 'Annee'}, inplace=True)
    st.write("Temps moyen par type d'alarme, district, et type de bien")
    blist = total['PropertyType'].unique()
    blist.sort()
    dlist = total['IncGeo_BoroughName'].unique()
    dlist.sort()
    alist = total['SpecialServiceType'].unique()
    alist.sort()
    district0 = st.selectbox("Select a district:",dlist, index=32)
    intervention = st.selectbox("Select a intervention",alist, index=19)
    endroit = st.selectbox("Select a type",blist, index=202)
    moyenne=total.loc[(total['IncGeo_BoroughName']==district0) & (total['SpecialServiceType']==intervention) & (total['PropertyType']==endroit),'AttendanceTimeSeconds'].mean()
    if np.isnan(moyenne):
        st.write("Pas d'intervention de ce type")
    else:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col2.metric("Moyenne :", int(total.loc[(total['IncGeo_BoroughName']==district0) & (total['SpecialServiceType']==intervention) & (total['PropertyType']==endroit),'AttendanceTimeSeconds'].mean()))
        col3.metric("En minutes :",int(moyenne/60))
        col4.metric("et secondes :", int(moyenne-(int(moyenne/60)*60)))
        col5.metric("Minimun : ", int(total.loc[(total['IncGeo_BoroughName']==district0) & (total['SpecialServiceType']==intervention) & (total['PropertyType']==endroit),'AttendanceTimeSeconds'].min()))
        col6.metric("Maximum : ", int(total.loc[(total['IncGeo_BoroughName']==district0) & (total['SpecialServiceType']==intervention) & (total['PropertyType']==endroit),'AttendanceTimeSeconds'].max()))
        col1.metric("Nombre d'intervention : ", len(total.loc[(total['IncGeo_BoroughName']==district0) & (total['SpecialServiceType']==intervention) & (total['PropertyType']==endroit)]))
        fig=px.box(total.loc[(total['IncGeo_BoroughName']==district0) & (total['SpecialServiceType']==intervention) & (total['PropertyType']==endroit),'AttendanceTimeSeconds'], y='AttendanceTimeSeconds', height=800, points="all")
        fig.update_layout(
        title="Temps d'attente en graphique",
           yaxis=dict(title="Temps d'attente (s)")
           )
        st.plotly_chart(fig, use_container_width=True)

elif selected=='Type':
    st.write("Groupe d'incident",pd.read_csv("IncidentGroup.csv"))
    st.write("Les fausses alarmes regroupe AFA, False alarm - Good intent et False alarm - Malicious")
    st.write("Les feux regroupe Primary Fire, Secondary Fire, Chimney Fire et Late Call")
    st.write("Description du type",pd.read_csv("StopCodeDescription.csv"))
    st.write("Types Speciaux")
    st.dataframe(pd.read_csv("SpecialServiceType.csv"))

else:
    st.title("Contexte")
    st.write("Ceci est la description du projet :")
    st.write("Le projet qui nous a été assigné est : Prédiction du temps de réponse d'un vehicule de la Brigade des Pompiers de Londres. (Vous pouvez trouver les jeux de données ici : https://data.london.gov.uk/dataset/london-fire-brigade-incident-records et https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records )")
    st.write("A Londres, il y a 102 brigades de pompiers sur les 33 districts. ( source : https://www.london-fire.gov.uk/community/your-borough/ )")
    col1, col2 = st.columns(2)
    with col1:
        st.write('Les districts de Londres')
        st.image('https://i.pinimg.com/736x/d7/c9/ae/d7c9aed78b880be11aaf154fa9e800a4--london-boroughs-map-it.jpg')#, width=350)
    with col2:
        st.write('Localisation des brigades des pompiers')
        st.image('https://raw.githubusercontent.com/FirePredict/fire_predict/main/Caserne.jpg')#, width=375)
