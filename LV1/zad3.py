numbers = []
while True:
    try:
        x = input()
        if x=="Done":
            break
        x = int(x)
        numbers.append(x)
    except:
        print("Please enter valid number!")

print("Number of input numbers: " + str(len(numbers)))
print("Min: "+str(min(numbers)))
print("Max: "+str(max(numbers)))

print("Avg: "+str(sum(numbers)/len(numbers)))