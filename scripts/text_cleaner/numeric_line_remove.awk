#!/usr/bin/awk -f


#Removing lines composed only by numbers 

match($0, /(^|[^[:alpha:]]{1,})[[:space:]][[:digit:]]{1,}([^[:alpha:]]{1,}|$)/) {
    str=substr($0, RSTART, RLENGTH)
    sub(str,"",$0 )

}1
