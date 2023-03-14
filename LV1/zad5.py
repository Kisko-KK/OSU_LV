f = open('SMSSpamCollection.txt')

spamCounter = 0
hamCOunter = 0
wordSpamCOunter = 0
wordHamCOunter = 0

spamSpecialCounter = 0
for line in f:
    line = line.rstrip()
    words = line.split()
    
    if words[0] == "spam":
        spamCounter += 1
        wordSpamCOunter -= 1
        for word in words:
            wordSpamCOunter +=1
        if words[len(words)-1].endswith("!"):
            spamSpecialCounter += 1
    else:
        hamCOunter += 1
        wordHamCOunter -= 1
        for word in words:
            wordHamCOunter +=1



print("Spam avg: " + str(wordSpamCOunter / spamCounter))
print("Ham avg: " + str(wordHamCOunter / hamCOunter))
print(spamSpecialCounter)




