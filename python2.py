import python1
a=int(input("Enter yout acc no: "))
for i in python1.person.keys():
    if a==int(i):
        word=python1.person[i]["name"].split()
        if len(word) < 2:
            print("Invalid name format. Please ensure it contains both first and last names.")
            break
        print("FName:",word[0])
        print("LName:",word[1])
        print("Balance:",python1.person[i]["balance"])
        break
    else:
        print("Account number not found.")
        break