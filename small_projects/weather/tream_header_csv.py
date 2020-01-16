in_file = 'data.csv'
out_file = 'tem10y.csv'

with open(in_file, 'rt', encoding = 'EUC_KR') as fr:
    lines = fr.readlines()

lines = ["연,월,일,기온,품질,균질\n"] + lines[5:]
lines = map(lambda v: v.replace('/', ','), lines)
result = ''.join(lines).strip()
print(result)

with open(out_file, 'wt', encoding='utf-8') as fw:
    fw.write(result)
    print('saved.')