# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:57:45 2023

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('API_19_DS2_en_csv_v2_5361599.csv')

# Filter the data for Urban Growth and Mortality rate)
df = data[data['Indicator Name'] == 'Urban population (% of total population)']

country =['Italy', 'Australia', 'Jamaica', 'China', 'United Arab Emirates',
             'Germany', 'France','Pakistan', 'Thailand']

c=df[df['Country Name'].isin(country)]

trans =c.transpose()

trans.rename(columns=trans.iloc[0],inplace=True)

Trans= trans.iloc[4:]

# Create a line plot for the data

plt.plot(Trans.index, Trans['Italy'], label='Italy')
plt.plot(Trans.index, Trans['Australia'], label='Australia', color='g')
plt.plot(Trans.index, Trans['Jamaica'], label='Jamaica')
plt.plot(Trans.index, Trans['China'], label='China')
plt.plot(Trans.index, Trans['United Arab Emirates'], label='United Arab Emirates')
plt.plot(Trans.index, Trans['Germany'], label='Germany')
plt.plot(Trans.index, Trans['France'], label='France')
plt.plot(Trans.index, Trans['Pakistan'], label='Pakistan')
plt.plot(Trans.index, Trans['Thailand'], label='Thailand')
plt.xlim('2010','2020')
plt.xlabel('Year')
plt.ylabel('Urban population growth (annual %)')
plt.title('Urban population growth (annual %)')
plt.legend()
plt.show()

plt.bar(Trans.index, Trans['Italy'], label='Italy')
plt.bar(Trans.index, Trans['Australia'], label='Australia')
plt.bar(Trans.index, Trans['Jamaica'], label='Jamaica')
plt.bar(Trans.index, Trans['China'], label='China')
plt.bar(Trans.index, Trans['United Arab Emirates'], label='United Arab Emirates')
plt.bar(Trans.index, Trans['Germany'], label='Germany')
plt.bar(Trans.index, Trans['France'], label='France')
plt.bar(Trans.index, Trans['Pakistan'], label='Pakistan')
plt.bar(Trans.index, Trans['Thailand'], label='Thailand')
plt.xlim('2010','2020')
plt.xlabel('Year')
plt.ylabel('Mortality rate, under-5 (per 1,000 live births)')
plt.title('Mortality rate, under-5 (per 1,000 live births)')
plt.legend()
plt.show()

# Filter the data for CO2 emissions from liquid fuel consumption (kt), Energy use,
gdp_pc_df = data[data['Indicator Name'] == 'Urban population']

country =['Italy', 'Australia', 'Jamaica', 'China', 'United Arab Emirates',
             'Germany', 'France','Pakistan', 'Thailand']

k = gdp_pc_df[gdp_pc_df['Country Name'].isin(country)]

trans =k.transpose()

trans.rename(columns=trans.iloc[0],inplace=True)

Trans= trans.iloc[4:]


plt.plot(Trans.index, Trans['Italy'], label='Italy')
plt.plot(Trans.index, Trans['Australia'], label='Australia', color='g')
plt.plot(Trans.index, Trans['Jamaica'], label='Jamaica')
plt.plot(Trans.index, Trans['China'], label='China')
plt.plot(Trans.index, Trans['United Arab Emirates'], label='United Arab Emirates')
plt.plot(Trans.index, Trans['Germany'], label='Germany')
plt.plot(Trans.index, Trans['France'], label='France')
plt.plot(Trans.index, Trans['Pakistan'], label='Pakistan')
plt.plot(Trans.index, Trans['Thailand'], label='Thailand')
plt.xlim('2010','2020')
plt.xlabel('Year')
plt.ylabel('CO2 emissions from liquid fuel consumption (kt)')
plt.title('CO2 emissions from liquid fuel consumption (kt)')
plt.legend()
plt.show()


plt.bar(Trans.index, Trans['Italy'], label='Italy')
plt.bar(Trans.index, Trans['Australia'], label='Australia')
plt.bar(Trans.index, Trans['Jamaica'], label='Jamaica')
plt.bar(Trans.index, Trans['China'], label='China')
plt.bar(Trans.index, Trans['United Arab Emirates'], label='United Arab Emirates')
plt.bar(Trans.index, Trans['Germany'], label='Germany')
plt.bar(Trans.index, Trans['France'], label='France')
plt.bar(Trans.index, Trans['Pakistan'], label='Pakistan')
plt.bar(Trans.index, Trans['Thailand'], label='Thailand')
plt.xlim('2010', '2020')
plt.xlabel('Year')
plt.ylabel('Urban population')
plt.title('Energy use (kg of oil equivalent per capita)')
plt.legend()
plt.show()



plt.scatter(Trans.index, Trans['Italy'], label='Italy')
plt.scatter(Trans.index, Trans['Australia'], label='Australia')
plt.scatter(Trans.index, Trans['China'], label='China')
plt.scatter(Trans.index, Trans['Jamaica'], label='Jamaica')
plt.scatter(Trans.index, Trans['United Arab Emirates'], label='United Arab Emirates')
plt.scatter(Trans.index, Trans['Germany'], label='Germany')
plt.scatter(Trans.index, Trans['France'], label='France')
plt.scatter(Trans.index, Trans['Pakistan'], label='Pakistan')
plt.scatter(Trans.index, Trans['Thailand'], label='Thailand')
plt.xlim('2010', '2020')
plt.xlabel('Year')
plt.ylabel('Arable land (% of land area)')
plt.title('Arable land (% of land area)')
plt.legend()
plt.show()