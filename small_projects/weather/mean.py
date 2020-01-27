import pandas as pd

df = pd.read_csv('export_dataframe.csv', encoding='utf-8')

'''
# 날짜별 기온 수집하기
md = {}
for i, row in df.iterrows():
    m, d, v = (int(row['월']), int(row['일']), float(row['온도']))
    key = str(m) + '/' + str(d)
    if key not in md:
        md[key] = []
    md[key] += [v]

# 날짜별 평균 구하기
avs = {}
for key in md:
    v = avs[key] = sum(md[key]) / len(md[key])
    print('{0} : {1}'.format(key,v))
'''

g = df.groupby(['월','일'])['온도']
gg = g.sum() / g.count()
print(gg)