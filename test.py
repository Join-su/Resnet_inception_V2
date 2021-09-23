f = open("class_sort2.txt", 'r', encoding="UTF-8")
line = f.readline()
dic_line = eval(line)
f.close()

print(dic_line)