x = []

while True :

    number = input("Enter a number: ")

    try :
        number = int(number)
        x.append(number)
    except :
        if number == "Done" :
          break
        else :
            print("Invalid input")


largest = None
largest = x[0]
for number in x :
    if number >= largest :
        largest = number
    else :
        continue
print(f"Maximum is {largest}")

smallest = None
smallest = x[0]
for number2 in x :
    if number2 <= smallest :
        smallest = number2
    else :
        continue
print(f"Minimum is {smallest}")