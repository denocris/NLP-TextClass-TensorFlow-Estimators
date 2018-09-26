#!/bin/sh

# DESCRIPTION
# This script performs a 'key search' filtering. i.e. filters
# from a corpus only rows that contains selected keys/words

# Usage:$: time bash key_search_v1.sh text_to_filter keys.txt

# remove old key_words_output
rm -rf key_count

text=$1
keysfile=$2

# list all the key words
#"(mese|settimana)\s.*(prossim[oa]|scors[oa])"
declare -a keys
readarray keys < $keysfile


init_wc=$(cat $text | wc -l)
echo "------ init_wc ---------" $init_wc
tot=0

## now loop through the key words
for key in "${keys[@]}"
do
   echo $key
   #echo $(cat $text | grep -w "$")
   rm -rf key_output
   cat $text | grep -vwP $key >> key_output
   rm -rf key_output_tmp
   out_wc=$(cat key_output | wc -l)
   num_sentences=$(($init_wc-$out_wc))
   echo "-------- num --------" $num_sentences
   echo -e $key '\t' $num_sentences >> key_count
   init_wc=$out_wc
   echo "-------- init_wc --------" $init_wc
   tot=$(($tot + $num_sentences))
   echo "-------- tot --------" $tot
   cp key_output key_output_tmp
   text=key_output_tmp
done

echo -e 'total' '\t' $tot >> key_count
