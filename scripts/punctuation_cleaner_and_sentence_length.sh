#!/bin/sh

# Usage:$: time bash script.sh <text_to_clean> <min_sentece_lenght> <max_sentece_lenght>
# Usage:$: time bash script.sh corpus.txt 6 16

# text to clean
text=$1
# parameter to select only sentence with more than N words
min_sentence_length=$2
# parameter to select only sentence with less than N words
max_sentence_length=$3

text_cleaner(){
          # substitute apostrophe "'" with a space
        	cat $text | sed "s/'/ /g" |
          # remove punctuations
        	tr -d '[:punct:]' |
          # select only sentence with more than N words
        	awk "NF>=$min_sentence_length" |
          # select only sentence with less than N words
          awk "NF<=$max_sentence_length" |
          # substitute upper letter with lower ones
        	tr [:upper:] [:lower:]
      }

text_cleaner $text >> corpus-cleaned-min$min_sentence_length-max$max_sentence_length
