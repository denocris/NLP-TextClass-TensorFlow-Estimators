#!/usr/bin/awk -f


#Removing uncomfortably long numbers

match($0, /[[:digit:]]{3,}/) {
    str=substr($0, RSTART, RLENGTH)
    sub(str,"",$0 )
}1
