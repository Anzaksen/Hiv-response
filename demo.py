import streamlit as st
import pandas as pd
import pickle
import joblib

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

st.set_page_config(layout="wide")

st.header('Hiv Treatment Response Predictor')
st.write('Select the following parameters to get your prediction')

FacilityType = ['Public', 'Faith Based', 'Private for profit', 'Private not for profit']
Sex = ['Female', 'Male']
MaritalStatus = ['Married', 'Single', 'Widowed', 'Seperated', 'Divorced', 'Cohabiting']
EducationLevel = ['Primary', 'Secondary', 'Tertiary', 'No Education', 'Others']
Occupation = ['Unemployed', 'Self employed', 'Business person', 'Civil servant', 'Student']
RegimenAtStart = ['TDF-3TC-EFV', 'AZT-3TC-NVP','AZT-3TC-EFV', 'TDF-3TC-NVP', 'TDF-FTC-EFV', 'ABC-3TC-EFV',
                    'TDF-FTC-NVP', 'TDF-3TC-LPV/r', 'TDF-3TC-ATV/r', 'ABC-3TC-LPV/r', 'TRV/EFV', 'AZT-3TC-TDF',
                    'AZT-3TC-ABC', 'TDF-FTC-ATV/r', 'ABC-3TC-NVP', 'ABC-3TC-ddi', 'ddi-3TC-NVP', 'AZT-3TC-ATV/r']
ClinicalStageAtStart = ['I', 'II', 'III', 'IV']
AdherenceCouncelingCompleted = ['Yes', 'No']
PregnancyStatus = ['Not Pregnant', 'Pregnant']

FacilityType = st.selectbox('Facility', options=FacilityType )
Age = st.slider('Age', 0, 100, 20)
sex = st.selectbox('Gender', options=Sex)
MaritalStatus = st.selectbox('Marital Status', options=MaritalStatus)
EducationLevel = st.selectbox('Level of Education', options=EducationLevel)
Occupation = st.selectbox('Occupation', options=Occupation)
RegimenAtStart = st.selectbox('Initial Regiment', options=RegimenAtStart)
WeightAtStart = st.slider('Initial Weight', 0, 100, 20)
HeightAtStart = st.slider('Initial Height', 0, 100, 20)
ClinicalStageAtStart = st.selectbox('Clinical Stage', options=ClinicalStageAtStart)
AdherenceCouncelingCompleted = st.selectbox('Did Adherence Counceling', options=AdherenceCouncelingCompleted)
PregnancyStatus = st.selectbox('Prgnancy Status', options=PregnancyStatus)
st.markdown('###')

data = {'FacilityType':FacilityType,
        'Age':Age,
        'Sex':sex,
        'MaritalStatus':MaritalStatus,
        'EducationLevel':EducationLevel,
        'Occupation':Occupation,
        'RegimenAtStart':RegimenAtStart,
        'WeightAtStart':WeightAtStart,
        'HeightAtStart':HeightAtStart,
        'ClinicalStageAtStart':ClinicalStageAtStart,
        'AdherenceCouncelingCompleted':AdherenceCouncelingCompleted,
        'PregnancyStatus':PregnancyStatus}
features = pd.DataFrame(data, index=[0])

le=joblib.load('labelencoders\FacilityTypeencoder.joblib')
features['FacilityType'] = le.transform(features['FacilityType'])

le=joblib.load('labelencoders\Sexencoder.joblib')
features['Sex'] = le.transform(features['Sex'])

le=joblib.load('labelencoders\MaritalStatusencoder.joblib')
features['MaritalStatus'] = le.transform(features['MaritalStatus'])

le=joblib.load('labelencoders\EducationLevelencoder.joblib')
features['EducationLevel'] = le.transform(features['EducationLevel'])

le=joblib.load('labelencoders\Occupationencoder.joblib')
features['Occupation'] = le.transform(features['Occupation'])

le=joblib.load('labelencoders\RegimenAtStartencoder.joblib')
features['RegimenAtStart'] = le.transform(features['RegimenAtStart'])

le=joblib.load('labelencoders\ClinicalStageAtStartencoder.joblib')
features['ClinicalStageAtStart'] = le.transform(features['ClinicalStageAtStart'])


le=joblib.load('labelencoders\AdherenceCouncelingCompletedencoder.joblib')
features['AdherenceCouncelingCompleted'] = le.transform(features['AdherenceCouncelingCompleted'])

le=joblib.load('labelencoders\PregnancyStatusencoder.joblib')
features['PregnancyStatus'] = le.transform(features['PregnancyStatus'])


st.subheader('CD4 Count Prediction')
GradientBoost = pickle.load(open('models\GradientBoost.pkl', 'rb'))
prediction = GradientBoost.predict(features)
AdaBoost = pickle.load(open('models\AdaBoost.pkl', 'rb'))
prediction_ada = AdaBoost.predict(features)
SVR = pickle.load(open('models\SVR.pkl', 'rb'))
prediction_svr = SVR.predict(features)
st.write(f'Using Gradient Boost Regressor: {round(prediction[0], 2)}')
st.write(f'Using Adaptive Boost Regressor: {round(prediction_ada[0], 2)}')
st.write(f'Using Support Vector Regressor: {round(prediction_svr[0], 2)}')
st.markdown('###')


st.subheader('Viral Count Prediction')
GradientBoost_viral = pickle.load(open('models\LinearRegression.pkl', 'rb'))
prediction_grad = GradientBoost_viral.predict(features)
Mlp = pickle.load(open('models\MLP.pkl', 'rb'))
prediction_mlp = Mlp.predict(features)
SVR_viral = pickle.load(open('models\SVR.pkl', 'rb'))
prediction_svrv = SVR_viral.predict(features)
st.write(f'Using Gradient Boost Regressor: {round(prediction_grad[0][0], 2)}')
st.write(f'Using Adaptive Boost Regressor: {round(prediction_mlp[0], 2)}')
st.write(f'Using Support Vector Regressor: {round(prediction_svrv[0], 2)}')