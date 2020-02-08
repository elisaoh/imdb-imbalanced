import os
import re

def get_only_chars(line):

    clean_line = ""

    line = line.lower()
    line = line.replace(" 's", " is") 
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.replace("'", "")
    line = line.replace("br","")
    

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    # print(clean_line)
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line



for dataset in ["test", "train"]:
	for label in ["pos","neg"]:
		folder = dataset +'/'+ label
		filenames = os.listdir(folder)
		with open(dataset+'_'+label+".csv",'a') as outfile:
			for fname in filenames:
				with open(folder+'/'+fname) as infile:
					for line in infile:
						good_line = get_only_chars(re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", line))
						outfile.write('"'+label+'"'+','+'"'+good_line+'"'+'\n')
		print("Finish preparing " + dataset +" " +label) 
