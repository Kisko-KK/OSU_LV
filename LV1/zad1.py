def total():
    return hours * euro

hours = int(input("Radni sati: "))
euro = float(input("Eura/h: "))

print("Ukupno: " + str(total()) + " eura.")