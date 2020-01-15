import urllib.request as req
import pandas as pd

url = 'https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv'
savefile = 'iris_test.csv'
req.urlretrieve(url, savefile)

csv = pd.read_csv(savefile, encoding='utf-8')
print(csv.head())