import pandas as pd
import numpy as np


excel_data = pd.read_excel('a.xlsx',sheet_name='sheet1')


#df =pd.DataFrame(columns=['MSRDT_DE', 'MSRSTE_NM', 'NO2', 'O3', 'CO','SO2','PM10','PM25'])
df = pd.DataFrame()

#print(excel_data['MSRDT_DE'])

#d = excel_data[(excel_data['MSRDT_DE'] == 20110101)]

#print(type(d.iloc[0]['MSRDT_DE'])) #numpy.int64
#print(type(d.iloc[0]['MSRSTE_NM'])) #str
#print(type(d.iloc[0]['NO2'])) #numpy.float64
#print(type(d.iloc[0]['O3'])) #numpy.float64
#print(type(d.iloc[0]['CO'])) #numpy.float64
#print(type(d.iloc[0]['SO2'])) #numpy.float64
#print(type(d.iloc[0]['PM10'])) #numpy.int64
#print(type(d.iloc[0]['PM25'])) #numpy.int64


columns=['MSRDT_DE', 'MSRSTE_NM', 'NO2', 'O3', 'CO','SO2','PM10','PM25']
excel_data[columns] = excel_data[columns].replace({'0':np.nan, 0:np.nan})

#print(excel_data.mean())

excel_data['NO2'] = excel_data['NO2'].fillna(excel_data['NO2'].mean())
excel_data['O3'] = excel_data['O3'].fillna(excel_data['O3'].mean())
excel_data['CO'] = excel_data['CO'].fillna(excel_data['CO'].mean())
excel_data['SO2'] = excel_data['SO2'].fillna(excel_data['SO2'].mean())
excel_data['PM10'] = excel_data['PM10'].fillna(excel_data['PM10'].mean())
excel_data['PM25'] = excel_data['PM25'].fillna(excel_data['PM25'].mean())





#if '강남구' not in d['MSRSTE']:
#   df = df.append({20110101,'강남구',평균들,...,평균}, ignore_index=True)
#print(d)
#df = df.append(d)
#print(df)


#print(df.iloc[0])

