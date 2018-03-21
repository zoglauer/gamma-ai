import Classifier

file_list = ["Ling.seq2.quality.root", "Ling.seq3.quality.root","Ling.seq4.quality.root"]
quality = ["Quality_seq2", "Quality_seq3", "Quality_seq4"]

index = 1

filename = file_list[1]
quality = quality[index]

runner = Classifier(filename, quality)