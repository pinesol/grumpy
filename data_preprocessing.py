import pandas as pd
import re
import time
import csv
from time import sleep
import numpy as np
from random import shuffle




def get_tagged_lyric(str_input):
    tagged_lyric = (str_input).replace('\r\n\r\n',' </l> </s> <s> <l> ')
    tagged_lyric = (tagged_lyric).replace('\r\n',' </l> <l> ')
    return '<start> <s> <l> ' + tagged_lyric + ' </l> </s>'


def replace_with_oov(input_str,vocab):
    result=''
    for word in input_str.split():
        if (word in vocab):
            result= result + word + ' '
        else:
            result= result + '<unk> '
    return result


def clean_str(string,tag):
    """
    Tokenization/string cleaning for each lyric set
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """


    if tag:
        string = get_tagged_lyric(string).lower()
    else:
        string = string.lower()

    string = re.sub(r"\([0-9]+x\)", "", string)
    string = re.sub(r"\[.*?\]", "", string)
    string = re.sub(r"\{.*?\}", "", string)
    string = re.sub(r"chorus", "", string)
    string = re.sub(r"verse", "", string)
    
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`<>/]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"<br />", " ", string) #Replace HTML break with white space
    string = re.sub(r"\\", " ", string)
    string = string.replace("<l> </l>","") # For empty lines    
    string = string.replace("  ","")
    string = string.replace("<s></s>","") # For empty stanzas
    string = string.replace("( "," (") # For empty stanzas
    return string.strip()



def main(tag=False):

	vocab_size = 10000


	print "Reading in lyrics from remote source..."
	pop = pd.read_csv('https://raw.githubusercontent.com/jamesthomson/Evolution_of_Pop_Lyrics/master/data/scraped_lyrics.tsv',sep='\t')


	# Remove rows that don't have Lyrics
	pop = pop[pop['lyrics']!='Lyrics Not found']


	x_text = [clean_str(sent,tag=tag) for sent in pop.lyrics]

    #x_text_tagged = [clean_str(sent,tag=True) for sent in pop.lyrics]

	print ("Building vocab...")

	word_count = {} # Keys are words, Values are frequency

	for review in x_text:

	    words = review.split()

	    for word in words:
	        try:
	            word_count[word]+=1
	        except:
	            word_count[word]=0


	res = list(sorted(word_count, key=word_count.__getitem__, reverse=True))

	#global vocab
	vocab = res[:vocab_size]

	# Replacing words that are not in the vocab with '<oov>'
	cleaned_x_text = [replace_with_oov(item,vocab) for item in x_text]

	#pop['Final_lyrics']=cleaned_x_text

	
	shuffle(cleaned_x_text)



	print ("Writing to files...")

	# Save cleaned and processed lyrics to as test, dev and train

	path_to_folder = "/Users/SeansMBP/Desktop/Cho/Project/data/not_tagged/"

	text_file = open(path_to_folder+"train.txt", "w")
	text_file.write('\n'.join(cleaned_x_text[:9447]))
	text_file.close()

	text_file = open(path_to_folder+"dev.txt", "w")
	text_file.write('\n'.join(cleaned_x_text[9447:12145]))
	text_file.close()

	text_file = open(path_to_folder+"test.txt", "w")
	text_file.write('\n'.join(cleaned_x_text[12145:]))
	text_file.close()


	print ("DONE!")


if __name__ == '__main__':
	main(tag=False)

