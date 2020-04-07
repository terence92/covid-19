import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import datetime
from datetime import date, timedelta
from sklearn.cluster import KMeans
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
import plotly.offline as py
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import streamlit as st

st.title('COVID 19')

st.subheader('SINTOMI COVID 19')
symptoms={'symptom':['Fever',
        'Dry cough',
        'Fatigue',
        'Sputum production',
        'Shortness of breath',
        'Muscle pain',
        'Sore throat',
        'Headache',
        'Chills',
        'Nausea or vomiting',
        'Nasal congestion',
        'Diarrhoea',
        'Haemoptysis',
        'Conjunctival congestion'],'percentage':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}

symptoms=pd.DataFrame(data=symptoms,index=range(14))
symptoms

fig = px.bar(symptoms[['symptom', 'percentage']].sort_values('percentage', ascending=False), 
             y="percentage", x="symptom", color='symptom', 
             log_y=True, template='ggplot2', title='Symptom of  Coronavirus')
fig.show()
plt.figure(figsize=(15,15))
plt.title('Symptoms of Coronavirus',fontsize=20)    
plt.pie(symptoms['percentage'],autopct='%1.1f%%')
plt.legend(symptoms['symptom'],loc='best')

st.pyplot()


# **Reading Data**

# %% [code]
data=pd.read_csv("https://raw.githubusercontent.com/terence92/covid-19/master/covid19-ita-regions.csv")

# %% [code]
an_data = pd.read_csv("https://raw.githubusercontent.com/terence92/covid-19/master/COVID19_open_line_list.csv")

# %% [code]
comp = pd.read_excel('https://github.com/terence92/covid-19/blob/master/COVID-19-3.27-top30-500.xlsx?raw=true')

# %% [code]
province = pd.read_csv("https://raw.githubusercontent.com/terence92/covid-19/master/covid19_italy_province.csv")

dat = pd.read_csv("https://raw.githubusercontent.com/terence92/covid-19/master/covid19_italy_province.csv")

# %% [markdown]
#  **Looking into data**

# %% [code]

# %% [code]
an_data = an_data[an_data['country']=='Italy']
an_data.shape
st.dataframe(an_data)

# %% [markdown]
# **Age distribution of Confirmation**
st.subheader('DISTRIBUZIONE ANNI CASI CONFERMATI')
# %% [code]
plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
plt.title("Age distribution of Confirmation")
sns.kdeplot(data=an_data['age'], shade=True).set(xlim=(0))
st.pyplot()


# %% [markdown]
# > **Age**
# 
# **Here, the graph shows the age distribution of the infected people by gender. We can clearly see older people are more likely to be infected, especially older people with having lung disease and problems in their respiratory system. The age group of 40 to 50yr are more infected than the rest of the population in men. On the other hand age groups of 50yr to 70yr are more infected in womens. As Dr.Steven Gambert, professor of medicine and director of geriatrics at the University of Maryland School of Medicine says “ Older people have higher risk of underlying health conditions, older people are already under physical stress, and their immune systems, even if not significantly compromised, simply do not have the same “ability to fight viruses and bacteria”. As data says Italy has the oldest population across globe by count. According to EU statistics Italy has the lowest percentage of young people**.

# %% [markdown]
# **Gender Distribution of Confirmatioin**
st.subheader('DISTRIBUZIONE DI GENERE COVID 19')
# %% [code]
plt.figure(figsize=(15, 5))
plt.title('Gender')
an_data.sex.value_counts().plot.bar();

# %% [code]
fig = px.pie( values=an_data.groupby(['sex']).size().values,names=an_data.groupby(['sex']).size().index)
fig.update_layout(
    font=dict(
        size=15,
        color="#242323"
    )
    )   
    
st.pyplot()





