import sys
import load_and_guess
import display_neural_activation

def DisplayMenu():
    print("1. Load image and guess value\n")
    print("2. Produce adversarial output(s) - input images must be placed into ./testdata\n")
    print("3. Display neural activation\n")
    print("4. Display this menu again\n")
    print("5. Quit\n")

def SwitchCase(userinput):
    if userinput == 1:
        fileinput = input("Enter file location: ")
        guess = load_and_guess.LAGmain(fileinput)
        print("Guess of original image: ", str(guess))
    if userinput == 2:
        exec(open("try_to_force.py").read(), locals(), locals())
    if userinput == 3:
        fileinput = input("Enter file location: ")
        display_neural_activation.DNAmain(fileinput)
    if userinput == 4:
        DisplayMenu()
 
print("***\nAdversarial Machine Learning Project - JUGGERNAUTS\n***\n")
DisplayMenu()

while 1:
    userinput = int(input("Select an option - 4 to display menu again: "))
    print("\n")
    if userinput == 5:
        break
    SwitchCase(userinput)
