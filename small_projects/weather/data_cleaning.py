import pandas as pd

data = pd.read_csv('weather_data.csv')
data = data.drop(['지점'], axis = 1)
data.columns = ['위치', '일시', '온도']
data['일시'] = pd.to_datetime(data['일시'], errors='coerce')
data['연'] = data['일시'].dt.year
data['월'] = data['일시'].dt.month
data['일'] = data['일시'].dt.day


data = data[['연', '월', '일', '온도', '위치']]


export_csv = data.to_csv (r'C:\Users\ody12\Desktop\export_dataframe.csv', index = None, header=True)

