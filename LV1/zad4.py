f = open('song.txt')
dict = dict()
for line in f:
    line = line.rstrip()
    words = line.split()
    for word in words:
        if word not in dict:
            dict[word] = 1
        else:
            dict[word] += 1



counter = 0
for key, val in dict.items():
    if val == 1:
        counter += 1
        print(key)

print(counter)
f.close()