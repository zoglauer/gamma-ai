from strippairing2 import StripPairing

flag = False

while(not flag):
    inVal = input("Would you like to run strippairing2?: ")
    if (inVal.lower() == 'exit'):
        print("Exiting the program...")
        flag = True
    elif (inVal == 'yes'):
        fileVal = input("Which file would you like to set? (['StripPairing.x2.y2.strippairing.root']): ")
        if fileVal:
            variable = input("How many Hidden Layers?" )
            runner = StripPairing(fileVal, variable)
            runner.run()
            flag = True
    # elif (inVal == 'CEZA'):
    #     fileVal = input("Which file would you like to set? (['Ling.seq2.quality.root', 'Ling.seq3.quality.root', 'Ling.seq4.quality.root']): ")
    #     qualityVal = input("Which quality would you like to set? (['Quality_seq2', 'Quality_seq3', 'Quality_seq4']): ")
    #     runner = CERA(fileVal, qualityVal)
    #     runner.run()
    #     flag = True