import numpy as np
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu("Menu", ["Contexte", 'Analyse', 'Graphs', 'Dataframe', 'Moyenne'], 
            icons=['house', 'activity', 'graph-up', 'table', 'calculator'], default_index=0)

total= pd.read_csv("limited.csv")

if selected == "Analyse":
    total.rename(columns={'CalYear_x': 'Annee'}, inplace=True)
    st.write("Temps moyen par mois par type d'alarme et par district")
    clist = total['IncGeo_BoroughName'].unique()
    alist = total['SpecialServiceType'].unique()
    district = st.selectbox("Select a district:",clist)
    intervention = st.selectbox("Select a intervention",alist)
    query = f"IncGeo_BoroughName=='{district}' & SpecialServiceType=='{intervention}'"
    df_filtered = total.query(query)
    st.write("Nb Intervention : ",len(df_filtered))
    if len(df_filtered)>0:
        fig = px.line(        
            df_filtered,
            x = "month", 
            y = "AttendanceTimeSeconds",
            title = "Temps D'attente"
        )
        fig = sns.relplot(x='month', y='AttendanceTimeSeconds',ci=None, height=14, kind='line', hue='Annee', data=df_filtered)
        plt.xlabel("Mois")
        plt.ylabel("Temps d'attente moyen (s)")
        st.pyplot(fig)
        st.write("Moyenne : ", int(total.loc[(total['IncGeo_BoroughName']==district) & (total['SpecialServiceType']==intervention),'AttendanceTimeSeconds'].mean()))
    else:
        st.write("Pas d'intervention de ce type")

elif selected == "Graphs":

    st.title("Les graphs :")
    option = st.selectbox('Quel graph ?', ('Nb_intervention_incident','Nb_intervention_batiment','Nb_intervention_district','Delai_district','Delai_incident','Delai_batiment','Delai_mois_annee'))
    if option == "Nb_intervention_incident":
        fig=px.histogram(total, x='SpecialServiceType', text_auto='.2s', height=600).update_xaxes(categoryorder='total descending')
        fig.update_layout(
                title="Nombre d'intervention par type d'incident",
                yaxis=dict(title="Nombre d'interventions"),
                xaxis=dict(title="Type d'intervention")
                )
        plt.xticks(rotation=90)
        st.plotly_chart(fig, use_container_width=True)

    if option == "Nb_intervention_batiment":
        fig=px.histogram(total, x='PropertyType', text_auto='.2s', height=600).update_xaxes(categoryorder='total descending')
        fig.update_layout(
                title="Nombre d'intervention par type de batiment",
                yaxis=dict(title="Nombre d'interventions"),
                xaxis=dict(title="Type de batiment")
                )
        plt.xticks(rotation=90)
        st.plotly_chart(fig, use_container_width=True)

    if option == "Nb_intervention_district":
        fig=px.histogram(total, x='IncGeo_BoroughName', text_auto='.2s', height=600).update_xaxes(categoryorder='total descending')
        fig.update_layout(
                title="Nombre d'intervention par district",
                yaxis=dict(title="Nombre d'interventions"),
                xaxis=dict(title="District")
                )
        plt.xticks(rotation=90)
        st.plotly_chart(fig, use_container_width=True)

    elif option=="Delai_district":

        fig=px.histogram(total, x='IncGeo_BoroughName', y='AttendanceTimeSeconds', histfunc='avg', height=600, text_auto='.0f').update_xaxes(categoryorder='total descending')
        fig.update_layout(
                title="Temps d'attente moyen par district",
                yaxis=dict(title="Temps d'attente (s)"),
                xaxis=dict(title="District")
                )
        plt.xticks(rotation=90)
        st.plotly_chart(fig, use_container_width=True)

    elif option=="Delai_incident":

        fig=px.histogram(total, x='SpecialServiceType', y='AttendanceTimeSeconds', histfunc='avg', height=600, text_auto='.0f').update_xaxes(categoryorder='total descending')
        fig.update_layout(
                title="Temps d'attente moyen par type d'incident",
                yaxis=dict(title="Temps d'attente (s)"),
                xaxis=dict(title="District")
                )
        plt.xticks(rotation=90)
        st.plotly_chart(fig, use_container_width=True)

    elif option=="Delai_batiment":
        fig=px.histogram(total, x='PropertyType', y='AttendanceTimeSeconds', histfunc='avg', height=600, text_auto='.0f').update_xaxes(categoryorder='total descending')
        fig.update_layout(
                title="Temps d'attente moyen par type de batiment",
                yaxis=dict(title="Temps d'attente (s)"),
                xaxis=dict(title="District")
                )
        plt.xticks(rotation=90)
        st.plotly_chart(fig, use_container_width=True)

    elif option=="Delai_mois_annee":
        total.rename(columns={'CalYear_x': 'Annee'}, inplace=True)
        st.write("Delai d'intervention par mois et annees")
        fig= sns.relplot(x='month', y='AttendanceTimeSeconds',ci=None, height=14, kind='line', hue = 'Annee', data=total)
        plt.xlabel("Mois")
        plt.ylabel("Temps d'attente moyen (s)")
        st.pyplot(fig)
        
elif selected == "Dataframe":

    st.title("Les 1000 premieres lignes :")
    st.dataframe(total.head(1000))

elif selected == "Moyenne":
    total.rename(columns={'CalYear_x': 'Annee'}, inplace=True)
    st.write("Temps moyen par type d'alarme, district, et type de batiment")
    blist = total['PropertyType'].unique()
    clist = total['IncGeo_BoroughName'].unique()
    alist = total['SpecialServiceType'].unique()
    district = st.selectbox("Select a district:",clist)
    intervention = st.selectbox("Select a intervention",alist)
    endroit = st.selectbox("Select a type",blist)
    
    moyenne=total.loc[(total['IncGeo_BoroughName']==district) & (total['SpecialServiceType']==intervention) & (total['PropertyType']==endroit),'AttendanceTimeSeconds'].mean()
    if np.isnan(moyenne):
        st.write("Pas d'intervention de ce type")
    else:
        st.write("Moyenne : ", int(total.loc[(total['IncGeo_BoroughName']==district) & (total['SpecialServiceType']==intervention) & (total['PropertyType']==endroit),'AttendanceTimeSeconds'].mean()), "soit : ",int(moyenne/60)," mn et ", int(moyenne-(int(moyenne/60)*60))," secondes")
        st.write("Min : ", int(total.loc[(total['IncGeo_BoroughName']==district) & (total['SpecialServiceType']==intervention) & (total['PropertyType']==endroit),'AttendanceTimeSeconds'].min()))
        st.write("Max : ", int(total.loc[(total['IncGeo_BoroughName']==district) & (total['SpecialServiceType']==intervention) & (total['PropertyType']==endroit),'AttendanceTimeSeconds'].max()))  
 
        fig=px.box(total.loc[(total['IncGeo_BoroughName']==district) & (total['SpecialServiceType']==intervention) & (total['PropertyType']==endroit),'AttendanceTimeSeconds'], y='AttendanceTimeSeconds', height=800, points="all")
        fig.update_layout(
        title="Temps d'attente en graphique",
           yaxis=dict(title="Temps d'attente (s)")
           )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.title("Contexte")
    st.write("Ceci est la description du projet :")
    st.write("Le projet qui nous a ete assigne est : Prediction du temps de reponse d'un vehicule de la Brigade des Pompiers de Londres. (Vous pourrez trouver les jeux de donnees ici : https://data.london.gov.uk/dataset/london-fire-brigade-incident-records et https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records )")
    st.write("A Londres, il y a 102 brigades de pompiers sur les 33 districts. ( source : https://www.london-fire.gov.uk/community/your-borough/ )")
    st.write("Les districts sont distribues ainsi :")
    st.image("https://i.pinimg.com/736x/d7/c9/ae/d7c9aed78b880be11aaf154fa9e800a4--london-boroughs-map-it.jpg" )
    st.write("Les brigades sont positionnees comme ceci :")
    st.image("https://raw.githubusercontent.com/FirePredict/fire_predict/main/Caserne.jpg")
    
