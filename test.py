while True:
    amount = input("Please enter the amount: ")
    try:
        amount = float(amount)
    except :
        amount = -1

    if amount < 0:
        print("Enter a valid amount")
    else:
        print(f"The amount is {amount}")
        break