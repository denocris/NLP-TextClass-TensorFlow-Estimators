#!/bin/sh

# DESCRIPTION
# This script performs a 'key search' filtering. i.e. filters
# from a corpus only rows that contains selected keys/words

# Usage:$: time bash key_search_v1.sh text_to_filter keys.txt

# remove old key_words_output
rm -rf key-output
rm -rf key-count

text=$1
keysfile=$2

# list all the key words
#"(mese|settimana)\s.*(prossim[oa]|scors[oa])"
declare -a keys
readarray keys < $keysfile


prec=0
tot=0
## now loop through the key words
for key in "${keys[@]}"
do
   echo $key
   cat $text | grep -wP $key >> key-output
   num_sentences=$(($(cat key-output | wc -l)-$prec))
   echo -e $key '\t' $num_sentences >> key-count
   prec=$(($prec + $num_sentences))
   tot=$(($tot + $num_sentences))
done

echo -e 'total' '\t' $tot >> key-count
