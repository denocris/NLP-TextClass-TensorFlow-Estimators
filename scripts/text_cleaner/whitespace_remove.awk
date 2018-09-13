#!/usr/bin/awk -f


#Removing excessive whitespaces 

match($0, /(([[:space:]]{2,})|(^[[:space:]]))/) {
    str=substr($0, RSTART, RLENGTH)
    sub(str,"",$0 )

}1
