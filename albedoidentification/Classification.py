from CERA import CERA
from CEZA import CEZA

flag = False

while(not flag):
    inVal = input("Would you like to run CERA or CEZA?: ")
    if (inVal.lower() == 'exit'):
        print("Exiting the program...")
        flag = True
    elif (inVal == 'CERA'):
        fileVal = input("Which file would you like to set? (['Ling.seq2.quality.root', 'Ling.seq3.quality.root', 'Ling.seq4.quality.root']): ")
        qualityVal = input("Which quality would you like to set? (['Quality_seq2', 'Quality_seq3', 'Quality_seq4']): ")
        runner = CEZA(fileVal, qualityVal)
        runner.run()
        flag = True
    elif (inVal == 'CEZA'):
        fileVal = input("Which file would you like to set? (['Ling.seq2.quality.root', 'Ling.seq3.quality.root', 'Ling.seq4.quality.root']): ")
        qualityVal = input("Which quality would you like to set? (['Quality_seq2', 'Quality_seq3', 'Quality_seq4']): ")
        runner = CERA(fileVal, qualityVal)
        runner.run()
        flag = True
