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

from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in symptoms.symptom)
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="white").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))

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


# %% [markdown]
# **Age distribution of Confirmation**

# %% [code]
st.subheader('DISTRIBUZIONE ANNI COVID 19')
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
    
#py.iplot(fig)
st.subheader('DISTRIBUZIONE DI GENERE - COVID 19')
st.pyplot()


# %% [markdown]
# **Age distribution of the confirmation by gender**

# %% [code]
male_dead = an_data[an_data.sex=='male']
female_dead = an_data[an_data.sex=='female']

# %% [code]
plt.figure(figsize=(10,6))
sns.set_style("darkgrid")
plt.title("Age distribution of the confirmation by gender")
sns.kdeplot(data=female_dead['age'], label="Women", shade=True).set(xlim=(0))
sns.kdeplot(data=male_dead['age'],label="Male" ,shade=True).set(xlim=(0))
st.pyplot()



# %% [code]
sns.set_style("whitegrid")
sns.FacetGrid(an_data,  size = 5)\
.map(plt.scatter, 'age', 'sex')\
.add_legend()
plt.title('Age vs Province',fontsize=40)
plt.xticks(fontsize=18)
plt.yticks(fontsize=28)

st.pyplot()

# %% [code]

# %% [markdown]
# **Checking for Null Value**

# %% [code]
data.isna().sum()

# %% [markdown]
# **Description of Data**

# %% [code]
data.describe().T

# %% [markdown]
# **Tracking the Patient**

# %% [code]
data.shape

# %% [code]
clus=data.loc[:,['SNo','Latitude','Longitude']]
clus.head()

# %% [markdown]
# **Checking for number of cluster**

# %% [code]
K_clusters = range(1,15)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = data[['Latitude']]
X_axis = data[['Longitude']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Score vs Cluster')
st.pyplot()

# %% [markdown]
# **The score get cosntant after 4 clusters, so making more clusters will not help us. The value for k is 4 in this case**

# %% [code]
kmeans = KMeans(n_clusters = 4, init ='k-means++')
kmeans.fit(clus[clus.columns[1:3]])
clus['cluster_label'] = kmeans.fit_predict(clus[clus.columns[1:3]])
centers = kmeans.cluster_centers_
labels = kmeans.predict(clus[clus.columns[1:3]])

# %% [markdown]
# **Graphical representation of clusters**

# %% [code]
clus.plot.scatter(x = 'Latitude', y = 'Longitude', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
st.pyplot()

# %% [markdown]
# **We will verify our clusters by putting values in world map by making use of folium library**

# %% [markdown]
# **Affected place in world map including Hospitalised , Confirm , Deaths and Recovery**

# %% [code]
import folium
italy_map = folium.Map(location=[42.8719,12.5674 ], zoom_start=5,tiles='Stamen Toner')

for lat, lon,RegionName,TotalPositiveCases,Recovered,Deaths,TotalHospitalizedPatients in zip(data['Latitude'], data['Longitude'],data['RegionName'],data['TotalPositiveCases'],data['Recovered'],data['Deaths'],data['TotalHospitalizedPatients']):
    folium.CircleMarker([lat, lon],
                        radius=5,
                        color='red',
                      popup =('RegionName: ' + str(RegionName) + '<br>'
                    'TotalPositiveCases: ' + str(TotalPositiveCases) + '<br>'
                    'TotalHospitalizedPatients: ' + str(TotalHospitalizedPatients) + '<br>'
                      'Recovered: ' + str(Recovered) + '<br>'
                      'Deaths: ' + str(Deaths) + '<br>'),

                        fill_color='red',
                        fill_opacity=0.7 ).add_to(italy_map)
italy_map
st.pyplot()

# %% [markdown]
# **The most affected cities and regions early in Italy are Lombardy, and then Emilia-Romagna, Veneto, Marche, and Piemonte. Milan is the second most populous Italian city which is located in Lombardy. Other areas in Italy which are affected by coronavirus include Toscana, Campania, Lazio, Liguria, Friuli Venezia Giulia, Sicilia, Puglia, Umbria, Abruzzo, Trento, Molise, Calabria, Sardegna, Valle d’Aosta, Basilicata, and Bolzano. As italy was the fourth most affected coronavirus country
# till last feb but now it has reached the maximum number of confirmed cases after China**

# %% [markdown]
# **Grouping Data According to  Region Name**

# %% [code]
data['Date'] = pd.to_datetime(data['Date']).dt.normalize()
daily = data.sort_values(['Date','Country','RegionName'])
latest = data[data.Date == daily.Date.max()]
latest.head()

# %% [code]
data_groupby_region = latest.groupby("RegionName")[['TotalPositiveCases', 'Deaths', 'Recovered','TestsPerformed','HospitalizedPatients','TotalHospitalizedPatients']].sum().reset_index()
dgr = data_groupby_region 
dgr.head()

# %% [markdown]
# **Desciption of Grouped Data by Region**

# %% [code]
dgr.describe().T

# %% [markdown]
# **Test performed vs Region**

# %% [code]
fig = px.bar(dgr[['RegionName', 'TestsPerformed']].sort_values('TestsPerformed', ascending=False), 
             y="TestsPerformed", x="RegionName", color='RegionName', 
             log_y=True, template='ggplot2', title='Test Performed vs Region')
st.pyplot()



# %% [markdown]
# **As the graph shows the test performed in different regions of Italy. Lombardia has the maximum number(25k+) of tests performed as it is the most infected in cities. As a result the next graph shows that it has the maximum number(7280) of positive coronavirus patients. Veneto is the second most infected city here followed by some more countries like Emilia Romagna, Lazio, Marche, Toscana, Piemonte, Friuli V.G. ,Campania, Sicilia, Liguria, Puglia, P.A. Trento, Calabria, Umbria, Abruzzo, Sardegna, Molisa, Basilicata, Valle d'Aosta, P.A. Bolzano etc.
# **

# %% [markdown]
# **Confirmed Cases vs Region**

# %% [code]
fig = px.bar(dgr[['RegionName', 'TotalPositiveCases']].sort_values('TotalPositiveCases', ascending=False), 
             y="TotalPositiveCases", x="RegionName", color='RegionName', 
             log_y=True, template='ggplot2', title='Confirmed Cases vs Region')
st.pyplot()



# %% [markdown]
# **There are more than 10,000 people who are infected through this virus. Italy is the most affected country in the world after China, with 827 deaths and 12,462 confirmed cases in almost three weeks. The government has restricted all flights from china. as because at the end of January after two Chinese tourists came down with coronavirus during a trip to Italy. At the time, it was hopefully the best measure which can
# block the spread of the disease**

# %% [markdown]
# **Hospitalised Patient vs Region**

# %% [code]
fig = px.bar(dgr[['RegionName', 'TotalHospitalizedPatients']].sort_values('TotalHospitalizedPatients', ascending=False), 
             y="TotalHospitalizedPatients", x="RegionName", color='RegionName', 
             log_y=True, template='ggplot2', title='Hospitalised Patient vs Region')
st.pyplot()


# %% [markdown]
# **Recovery vs Region**

# %% [code]
fig = px.bar(dgr[['RegionName', 'Recovered']].sort_values('Recovered', ascending=False), 
             y="Recovered", x="RegionName", color='RegionName', 
             log_y=True, template='ggplot2', title='Revovery vs Region')
st.pyplot()


# %% [markdown]
# **Death vs Region Name**

# %% [code]
fig = px.bar(dgr[['RegionName', 'Deaths']].sort_values('Deaths', ascending=False), 
             y="Deaths", x="RegionName", color='RegionName', 
             log_y=True, template='ggplot2', title='Death vs Region')
st.pyplot()


# %% [code]
dgrs_el = dgr.sort_values(by=['TotalPositiveCases'],ascending = False)
dgrs_el.head()

# %% [markdown]
# **Test and Confirm vs Region**

# %% [code]
plt.figure(figsize=(23,10))
plt.bar(dgrs_el.RegionName, dgrs_el.TestsPerformed,label="Tests Performed")
plt.bar(dgrs_el.RegionName, dgrs_el.TotalPositiveCases,label="Confirm Cases")
plt.xlabel('Region')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Test and Confirm vs Region',fontsize = 35)

st.pyplot()

f, ax = plt.subplots(figsize=(80,30))
ax=sns.scatterplot(x="RegionName", y="TestsPerformed", data=dgrs_el,
             color="red",label = "Tests Performed")
ax=sns.scatterplot(x="RegionName", y="TotalPositiveCases", data=dgrs_el,
             color="blue",label = "Confirm Cases")
ax.xaxis.set_tick_params(labelsize=35)

plt.plot(dgrs_el.RegionName,dgrs_el.TestsPerformed,zorder=1,color="red")
plt.plot(dgrs_el.RegionName,dgrs_el.TotalPositiveCases,zorder=1,color="blue")
st.pyplot()

# %% [markdown]
# **Confirmed cases vs People Hospitalised**

# %% [code]
plt.figure(figsize=(23,10))
plt.bar(dgrs_el.RegionName, dgrs_el.TotalPositiveCases,label="Confirm Cases")
plt.bar(dgrs_el.RegionName, dgrs_el.TotalHospitalizedPatients,label="Hospitalized Patients")

plt.xlabel('Region')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Confirm Cases vs People Hospitalised',fontsize= 35)
st.pyplot()

f, ax = plt.subplots(figsize=(40,20))

ax=sns.scatterplot(x="RegionName", y="TotalPositiveCases", data=dgrs_el,
             color="blue",label = "Confirm Cases")
ax=sns.scatterplot(x="RegionName", y="TotalHospitalizedPatients", data=dgrs_el,
             color="red",label = "Hospitalized Patients")
ax.xaxis.set_tick_params(labelsize=18)
plt.plot(dgrs_el.RegionName,dgrs_el.TotalPositiveCases,zorder=1,color="blue")
plt.plot(dgrs_el.RegionName,dgrs_el.TotalHospitalizedPatients,zorder=1,color="red")


# %% [markdown]
# **The graph shows statistical data direct from WHO. as the data says in Lombardia after 7,000 and more confirmed cases there are only approximately 4.5K people who are hospitalised. This has become a situation of crisis in italy. Hospital condition is becoming worse day by day. According to the doctors not every patient is getting proper and equal care and that is the main cause of multi fold spread of coronavirus. The whole country is locked down. Government has announced there will be no gathering, no sporting event and no travelling across the country just because of the high number of deaths in the country**

# %% [markdown]
# **Death and Recovery vs Region**

# %% [code]
plt.figure(figsize=(23,10))
plt.bar(dgrs_el.RegionName, dgrs_el.Recovered,label="Recovery")
plt.bar(dgrs_el.RegionName, dgrs_el.Deaths,label="Death")
plt.xlabel('Region')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Death and Recovery vs Region', fontsize= 35)
st.pyplot()

f, ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="RegionName", y="Recovered", data=dgrs_el,
             color="red",label = "Recovered")
ax=sns.scatterplot(x="RegionName", y="Deaths", data=dgrs_el,
             color="blue",label = "Deaths")
plt.plot(dgrs_el.RegionName,dgrs_el.Recovered,zorder=1,color="red")
plt.plot(dgrs_el.RegionName,dgrs_el.Deaths,zorder=1,color="blue")
st.pyplot()

# %% [markdown]
# **According to the graph recovery rate of the patient is very slow. There are some common reasons behind the rapid increase in numbers of people infected through coronavirus. According to the data, the number of hospitalized people is far less than the number of people infected through novel-Coronavirus. According to the geographical structure of Italy In Europe the cases have now been confirmed in every member nation of the European Union. Italy will remain totally locked down as its healthcare system struggles to cope, on the other hand the nearby countries like Germany and France report alarming spikes in daily cases**

# %% [code]
data['Date'] = pd.to_datetime(data['Date']).dt.normalize()
latest = data[data.Date == daily.Date.max()]

# %% [code]
temp = latest.loc[:,['Date','HospitalizedPatients','IntensiveCarePatients','TotalHospitalizedPatients','HomeConfinement','Recovered','Deaths','TotalPositiveCases','TestsPerformed']]
temp.head()

# %% [markdown]
# **Grouped by Provicne**

# %% [code]
province.head()

# %% [code]
provincegrp = province.groupby("ProvinceName")[["TotalPositiveCases"]].max().reset_index()

# %% [code]
fig = px.bar(provincegrp[['ProvinceName', 'TotalPositiveCases']].sort_values('TotalPositiveCases', ascending=False), 
             y="TotalPositiveCases", x="ProvinceName", color='ProvinceName', 
             log_y=True, template='ggplot2', title='Province vs Region')
st.pyplot()

# %% [code]
sns.set_style("whitegrid")
sns.FacetGrid(province,  size = 30)\
.map(plt.scatter, 'RegionName', 'ProvinceName')\
.add_legend()
plt.title('Age vs Infection Reason',fontsize=30)
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)
st.pyplot()

# %% [markdown]
# **Descipiton of Data grouped by Date**

# %% [code]
temp.describe().T

# %% [code]
data_groupby_date = latest.groupby("Date")[['Date','HospitalizedPatients','IntensiveCarePatients','TotalHospitalizedPatients','HomeConfinement','Recovered','Deaths','TotalPositiveCases','TestsPerformed']].sum().reset_index()
data_groupby_date

# %% [markdown]
# **Ratio and percentage of Confirmation, Deaths and Deaths, Recovery after Confirmation**

# %% [code]
ps_ts = float(data_groupby_date.TotalPositiveCases/data_groupby_date.TestsPerformed)
d_ts = float(data_groupby_date.Deaths/data_groupby_date.TestsPerformed)
r_ps = float(data_groupby_date.Recovered/data_groupby_date.TotalPositiveCases)
d_ps = float(data_groupby_date.Deaths/data_groupby_date.TotalPositiveCases)

# %% [code]
print("The percentage of Confirmation is "+ str(ps_ts*100) )
print("The percentage of Death is "+ str(d_ts*100) )
print("The percentage of Death after confirmation is "+ str(d_ps*100) )
print("The percentage of recovery after confirmation is "+ str(r_ps*100) )

# %% [code]
data_groupby_date1 = data.groupby("Date")[['TotalPositiveCases', 'Deaths', 'Recovered','TestsPerformed','HospitalizedPatients','TotalHospitalizedPatients']].sum().reset_index()
dgd3 = data_groupby_date1
dgd3.head()

# %% [code]
dgd2 = dgd3

# %% [code]
dgd2["Date"]= dgd3["Date"].dt.strftime("%d-%m-%y") 
dgd2.head()

# %% [markdown]
# **Test vs Confirmed**
# 

# %% [code]
dgd2 = dgd2.tail(14)

# %% [code]
plt.figure(figsize=(23,10))
plt.bar(dgd2.Date, dgd2.TestsPerformed,label="Tests Performed")
plt.bar(dgd2.Date, dgd2.TotalPositiveCases,label="Confirm Cases")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Test Peroformed vs Confirmed Cases',fontsize = 35)
st.pyplot()

f, ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="Date", y="TestsPerformed", data=dgd2,
             color="red",label = "Tests Performed")
ax=sns.scatterplot(x="Date", y="TotalPositiveCases", data=dgd2,
             color="blue",label = "Confirm Cases")
plt.plot(dgd2.Date,dgd2.TestsPerformed,zorder=1,color="red")
plt.plot(dgd2.Date,dgd2.TotalPositiveCases,zorder=1,color="blue")

# %% [markdown]
# **Confirmed cases vs People Hospitalised**

# %% [code]
plt.figure(figsize=(23,10))
plt.bar(dgd2.Date, dgd2.TotalPositiveCases,label="Confirm Cases")
plt.bar(dgd2.Date, dgd2.TotalHospitalizedPatients,label="Hospitalized Patients")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Confirmed Cases vs Hospitalised Cases',fontsize= 35)
st.pyplot()

f, ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="Date", y="TotalHospitalizedPatients", data=dgd2,
             color="red",label = "Hospitalized Patients")
ax=sns.scatterplot(x="Date", y="TotalPositiveCases", data=dgd2,
             color="blue",label = "Confirm Cases")
plt.plot(dgd2.Date,dgd2.TotalHospitalizedPatients,zorder=1,color="red")
plt.plot(dgd2.Date,dgd2.TotalPositiveCases,zorder=1,color="blue")

# %% [markdown]
# **Hospitalise vs Recevery and Death**

# %% [code]
plt.figure(figsize=(23,10))
plt.bar(dgd2.Date, dgd2.TotalHospitalizedPatients,label="Hospitaise Patients")
plt.bar(dgd2.Date, dgd2.Recovered,label="Recovery")
plt.bar(dgd2.Date, dgd2.Deaths,label="Death")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Hospitalise vs Recovery vs Death',fontsize=30)
st.pyplot()

f, ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="Date", y="TotalHospitalizedPatients", data=dgd2,
             color="black",label = "Hospitalise Patients")
ax=sns.scatterplot(x="Date", y="Recovered", data=dgd2,
             color="red",label = "Recovery")
ax=sns.scatterplot(x="Date", y="Deaths", data=dgd2,
             color="blue",label = "Death")
plt.plot(dgd2.Date,dgd2.TotalHospitalizedPatients,zorder=1,color="black")
plt.plot(dgd2.Date,dgd2.Recovered,zorder=1,color="red")
plt.plot(dgd2.Date,dgd2.Deaths,zorder=1,color="blue")

# %% [markdown]
# **Confirm vs Recovery vs Death**

# %% [code]
plt.figure(figsize=(23,10))
plt.bar(dgd2.Date, dgd2.TotalPositiveCases,label="Confirm")
plt.bar(dgd2.Date, dgd2.Recovered,label="Recovery")
plt.bar(dgd2.Date, dgd2.Deaths,label="Death")
plt.xlabel('Date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Confrim vs Recovery vs Death',fontsize=30)
st.pyplot()

f, ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="Date", y="TotalPositiveCases", data=dgd2,
             color="black",label = "Confirm")
ax=sns.scatterplot(x="Date", y="Recovered", data=dgd2,
             color="red",label = "Recovery")
ax=sns.scatterplot(x="Date", y="Deaths", data=dgd2,
             color="blue",label = "Death")
plt.plot(dgd2.Date,dgd2.TotalPositiveCases,zorder=1,color="black")
plt.plot(dgd2.Date,dgd2.Recovered,zorder=1,color="red")
plt.plot(dgd2.Date,dgd2.Deaths,zorder=1,color="blue")

# %% [markdown]
# **This graph gives an overview of the current situation of italy. There are more than 12,000 confirmed cases now. There is approximately equal number of deaths as of recovery. From the date the country has confirmed its first case of positive coronavirus it has been increasing exponentially. Till the date 11,Mar italy has become the second most infected country after China.**

# %% [code]
data_groupby_date1 = data.groupby("Date")[['TotalPositiveCases', 'Deaths', 'Recovered','TestsPerformed','HospitalizedPatients','TotalHospitalizedPatients']].sum().reset_index()
dgd1 = data_groupby_date1
dgd1.head()

