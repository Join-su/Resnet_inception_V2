import pandas as pd

f = open("100_rate.txt", 'r', encoding="UTF-8")
line = f.readline()
dic_line = eval(line)
f.close()

print(type(dic_line))
print(dic_line)
print(dic_line.keys())
print(list(dic_line))
class_list = list(dic_line)
print(class_list)
class_list.sort()
print(class_list)

df = pd.DataFrame.from_dict(dic_line, orient='index')  # transpose to look just like the sheet above
df.to_csv('100_text.csv')
# df.to_excel('file.xls')