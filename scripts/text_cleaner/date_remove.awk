#!/usr/bin/awk -f


#Removing dates with formats dd mmm yyyy and variations accordin to regex parameters

match($0, /(^|[^[:alpha:]])[[:digit:]]{2}[[:space:]]{1,}[[:alpha:]]{3,8}[[:space:]]{1,}[[:digit:]]{4}([^[:alpha:]]|$)/) {
    str=substr($0, RSTART, RLENGTH)
    sub(str," ",$0 )
}1
