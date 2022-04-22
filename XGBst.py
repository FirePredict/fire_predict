#!pip install xgboost --index-url https://artifactory.tech.orange/artifactory/api/pypi/pythonproxy/simple
#!pip install lime --index-url https://artifactory.tech.orange/artifactory/api/pypi/pythonproxy/simple
#!pip install streamlit --index-url https://artifactory.tech.orange/artifactory/api/pypi/pythonproxy/simple
    
import lime.lime_tabular  
import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from time import time

#visualisations
import matplotlib.pyplot as plt
import seaborn as sns

#préprocessing ML
from sklearn.preprocessing import StandardScaler

#modèle de ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from streamlit.legacy_caching.hashing import _CodeHasher
from streamlit.script_run_context import get_script_run_ctx as get_report_ctx
from streamlit.server.server import Server

st.set_page_config(layout="wide")
def main():
    state = _get_state()
    pages = {
        "Pré-processing": page_preprocessing,
        "Statistiques (Loi normale ?)": page_regression,
        "Machine Learning": page_machineLearning
    }
    
    with st.sidebar:
        page = option_menu("Menu", tuple(pages.keys()),
        #page = st.sidebar.radio("____________________________________________", tuple(pages.keys())),
        icons = ['activity','graph-up','gear'], default_index=0)

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

#with st.sidebar:
#    selected = option_menu("Menu", ["Pré-processing", "Régréssion - Loi normale", "Machine Learning"],
#    icons = ['activity','graph-up','gear'], default_index=0)

def page_preprocessing(state):
    st.title("Pré-processing")
    
    with st.spinner('Calculating...'):
        # Chargement des fichiers de données
        mobilisation = pd.read_csv("LFB Mobilisation data Last 3 years.csv", header=0, sep=";")
        incident = pd.read_csv("LFB Incident data Last 3 years.csv", header=0, sep=";")

        # Merge des 2 datasets
        total = pd.merge(incident, mobilisation, on='IncidentNumber')
        
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Valeurs NaN existantes par variable :"):
                    # Graphe qui montre les valeurs null
                    null_df = total.apply(lambda x: sum(x.isnull())).to_frame(name='count')
                    fig = plt.figure(figsize=(15, 5))

                    plt.plot(null_df.index, null_df['count'])
                    plt.xticks(null_df.index, null_df.index, rotation=45,horizontalalignment='right')

                    plt.xlabel('column names')
                    plt.margins(0.1)

                    #plt.show()
                    st.pyplot(fig)

        with col2:
            with st.expander("Valeurs NaN aprés la suppression :"):
                    #Suppression des valeurs Nan
                    total = total.dropna(axis = 1)

                    #Vérification des valeurs Nan
                    null_df = total.apply(lambda x: sum(x.isnull())).to_frame(name='count')
                    fig = plt.figure(figsize=(15, 5))

                    plt.plot(null_df.index, null_df['count'])
                    plt.xticks(null_df.index, null_df.index, rotation=45,horizontalalignment='right')

                    plt.xlabel('column names')
                    plt.margins(0.1)

                    #plt.show()
                    st.pyplot(fig)

        # Ajout d'une nouvelle variable pour la prédiction à la minute
        total['minute'] = total['AttendanceTimeSeconds']/60
        total['minute'] =  total['minute'].astype('int64', copy=False)

        #Separation du dataframe en 2 catégorie "variables catégorielles" et "variables qualitatives"
        numerical_cols = [contname for contname in total.columns if total[contname].dtype in ['float64', 'int64']]
        total_numerical = total[numerical_cols]

        qualitative_cols = [contname for contname in total.columns if total[contname].dtype in ['object']]
        total_qualitative = total[qualitative_cols]
        
        with st.expander("Choix des variables :"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("Nombre de modalités par variables")
                fig = plt.figure(figsize=(15, 10))
                sns.heatmap(data=total_qualitative.nunique().to_frame(), cmap='RdBu_r',fmt='d',annot=True).set_title('Nombre de valeurs unique par colonne:');
                st.pyplot(fig)

                # Suppression des modalités trop importantes et qui ne représente pas une notion métier capitale
                total_qualitative = total_qualitative.drop(["IncidentNumber", "Postcode_district", "DateOfCall", "TimeOfCall", "UPRN", "FRS", "DateAndTimeMobilised", "DateAndTimeArrived", "PlusCode_Code", "PlusCode_Description"], axis = 1)
            with col2:
                st.write("Nombre de modalités aprés suppression de certaines variables")
                fig = plt.figure(figsize=(15, 10))
                sns.heatmap(data=total_qualitative.nunique().to_frame(), cmap='RdBu_r',fmt='d',annot=True).set_title('Nombre de valeurs unique par colonne:');
                st.pyplot(fig)

                # Discrétisation des modalités
                total_qualitative = pd.get_dummies(total_qualitative)

            # Vérification de la dimension des données
            st.write("Dimensions du modèle de donnée aprés la discrétisation : ")
            st.write(total_qualitative.shape)

            # Merge des 2 datasets "indicatrices" et "qualitatives"
            df_merge = pd.concat([total_numerical, total_qualitative], axis = 1)
            state.total = total
            state.df_merge = df_merge

def page_regression(state):
        #Description statistiques pour démontrer que le modèle respecte une loi normale
        st.title("Statistiques (Loi normale ?)")

        with st.spinner('Calculating...'):
            st.write(state.df_merge.describe())
            
            with st.expander("Graphes :"):
                # Graphe de répartition moyenne = médiane
                num_col = state.df_merge._get_numeric_data().columns
                describe_num_df = state.df_merge.describe(include=['int64','float64'])
                describe_num_df.reset_index(inplace=True)                  
                describe_num_df = describe_num_df[describe_num_df['index'] != 'count']

                col1, col2 = st.columns(2)
            
                with col1:
                    n = 1
                    for i in num_col:  
                        if i in ['index']:    
                            continue  

                        if i in ['IncidentGroup_False Alarm'] or n == 8:
                            break
                            
                        fig = sns.factorplot(x="index", y=i, data=describe_num_df,size=5, aspect=2)
                        
                        #plt.show()
                        st.pyplot(fig)
                        n = n+1
                        
                with col2:
                    n = 1
                    for i in num_col:  
                        if i in ['index'] or n < 8:  
                            n = n+1
                            continue  

                        if i in ['IncidentGroup_False Alarm']:
                            break
                            
                        fig = sns.factorplot(x="index", y=i, data=describe_num_df,size=5, aspect=2)
                        
                        #plt.show()
                        st.pyplot(fig)

def page_machineLearning(state):
        st.title("Machine Learning")
    
        # Création des listes de selection
        parameters = ""

        alist = state.total['IncGeo_BoroughName'].unique()
        ListeDistrict = st.multiselect("Select a district:",alist,default=["WESTMINSTER","KENSINGTON AND CHELSEA"])

        for x in ListeDistrict:
            parameters = parameters + "IncGeo_BoroughName_" + x + ','

        #st.write(parameters)

        blist = state.total['IncidentGroup'].unique()
        ListeIncidentGroup = st.multiselect("Select a Incident Group:",blist,default=["Special Service","Fire", "False Alarm"])

        for x in ListeIncidentGroup:
            parameters = parameters + "IncidentGroup_" + x + ','

        #st.write(parameters)

        clist = state.total['IncidentStationGround'].unique()
        listeIncidentStation = st.multiselect("Select a Incident Station:",clist,default=["Soho","Paddington"])

        for x in listeIncidentStation:
            parameters = parameters + "IncidentStationGround_" + x + ','

        parameters = parameters[:-1]
        parameters = list(parameters.split(','))

        values = ['XGBoost','RandomForest','LinearRegression']
        mlEngine = st.selectbox("Select a ML Method:",values,index=values.index("XGBoost"))

        t0 = time()

        with st.expander("Résultat :"):
            #Moteur du machine learning
            with st.spinner('Calculating...'):
                if parameters:
                    if parameters[0] != '':
                        if mlEngine != '':
                            # Séparation de la variable cible du reste des données avec un ratio de 80 - 20 %
                            y = state.df_merge.minute
                            X = state.df_merge[parameters]

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=101)

                            # Initialisation du modèle
                            if mlEngine == 'RandomForest':
                                xgbr = RandomForestRegressor(n_estimators = 100, random_state = 2022)
                            elif mlEngine == 'XGBoost':
                                xgbr = xgb.XGBRegressor(verbosity=0) 
                            else:
                                xgbr = LinearRegression()

                            # Entrainement du modèle
                            xgbr.fit(X_train, y_train)

                            # Score du modèle sur les données d'entrainements et de test
                            #print("Training score R²: ", xgbr.score(X_train, y_train))
                            #print("Test score R²: ", xgbr.score(X_test, y_test))

                            # Resultat de prédiction 
                            y_pred = xgbr.predict(X_test)

                            col1, col2 = st.columns(2)
                            with col1:
                                # Graphe des variables les plus importantes qui ont influé sur le modèle
                                st.write("Les variables, les plus importantes, qui ont influées sur le modèle d'apprentissage")

                                fig = plt.figure(figsize=(10, 10))

                                if mlEngine == 'LinearRegression':
                                    sorted_idx = xgbr.coef_.argsort()
                                    plt.barh(X.columns, xgbr.coef_[sorted_idx][:10])
                                    plt.xlabel("LinearRegression Feature Importance")
                                elif mlEngine == 'XGBoost':
                                    sorted_idx = xgbr.feature_importances_.argsort()
                                    plt.barh(X.columns, xgbr.feature_importances_[sorted_idx][:10])
                                    plt.xlabel("Xgboost Feature Importance")
                                else:
                                    sorted_idx = xgbr.feature_importances_.argsort()
                                    plt.barh(X.columns, xgbr.feature_importances_[sorted_idx][:10])
                                    plt.xlabel("RandomForestRegression Feature Importance")

                                #plot_importance(xgbr)
                                #plt.show()
                                st.pyplot(fig)
                            
                            with col2:
                                st.write("Intérprétation des résultats")
                                st.write("             ")
                                t1 = time() - t0
                                st.metric("Calcul réalisé ", value="en {} secondes".format(round(t1,3)))
                                explainer = lime.lime_tabular.LimeTabularExplainer(training_data=np.array(X_test), ## 2d array,
                                                     feature_names=X_test.columns, 
                                                     mode='regression',
                                                     verbose=True,
                                                     random_state= 1)### la prédiction se base sur un modèle linéaire
                                lime_results = explainer.explain_instance(X_test.iloc[0], 
                                                                      xgbr.predict ,# fonction de prediction
                                                                     num_features=10
                                                                    )  ## le nombre maximale de variables présents dans l'explication
                                with plt.style.context("ggplot"):
                                    st.pyplot(lime_results.as_pyplot_figure())

                                st.metric("Score prédit : ", y_pred[0])
                                st.metric("La valeur actuelle :     ", y_test.iloc[0])

                            #lime_results.show_in_notebook(show_table=True, show_all=False)
                            components.html(lime_results.as_html(), height=800)
                    
                    
class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()