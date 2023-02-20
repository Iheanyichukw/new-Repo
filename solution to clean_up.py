import clean_up as cl

import numpy as np 

import pandas as pd

data = open("big_data.txt")
print (data)

all_words = []
counter = 0 
for line in data: 
    words = line.split()
    counter = counter + 1
    for word in words:
        word = cl.clean(word)
        all_words.append(word)
print (all_words)
        

df_words = pd.DataFrame(all_words, columns =('words',))

df_counts= df_words['words'].value_counts()

df_counts.to_csv('word_counts.csv')

print (df_words.count())