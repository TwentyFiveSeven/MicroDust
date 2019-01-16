import datetime
import requests
import json
import pandas as pd

startDate = datetime.date(2011,1,1)
#endDate = datetime.date(2018,12,31)

nowDate = startDate

openKey = '6f58494c7963393536356d7a595474'

Gu = ['강남구','강동구','강북구','강서구','관악구','광진구','구로구','금천구','노원구','도봉구','동대문구','동작구','마포구','서대문구','서초구','성동구','성북구','송파구','양천구','영등포구','용산구','은평구','종로구','중구','중랑구']


df =pd.DataFrame(columns=['MSRDT_DE', 'MSRSTE_NM', 'NO2', 'O3', 'CO','SO2','PM10','PM25'])

while True:
    date2String = nowDate.strftime('%Y%m%d')
    
    if date2String == '20190101':
        break
    
    for gu in Gu:
        url = 'http://openapi.seoul.go.kr:8088/'+openKey+'/json/DailyAverageAirQuality/1/1/'+date2String+'/'+gu
        response = requests.get(url).json()
        if 'DailyAverageAirQuality' in response:
            d = response['DailyAverageAirQuality']['row'][0]
        else:
            d = {'MSRDT_DE': date2String, 'MSRSTE_NM': gu, 'NO2': 0.0, 'O3': 0.0, 'CO': 0.0, 'SO2': 0.0, 'PM10': 0.0, 'PM25': 0.0}
        df = df.append(d,ignore_index=True)

    nowDate = nowDate + datetime.timedelta(days=1)

df.to_excel('b.xlsx', sheet_name='Sheet1')


