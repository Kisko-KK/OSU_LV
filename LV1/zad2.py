try:
    x = float(input())
    if x<=1 and x>=0:
        if x >= 0.9:
            print("A")
        elif x >= 0.8:
            print("B")
        elif x >= 0.7:
            print("C")
        elif x >= 0.6:
            print("D")
        else:
            print("F")
    else:
        print("Input number 0 - 1.0")
except:
    print("Something went wrong!")
