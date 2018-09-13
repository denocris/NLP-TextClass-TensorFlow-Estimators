#!/usr/bin/awk -f


#Removing numbers at the begin of a line

match($0, /^[[:digit:]]{1,}/) {
    str=substr($0, RSTART, RLENGTH)
    sub(str,"",$0 )
}1
