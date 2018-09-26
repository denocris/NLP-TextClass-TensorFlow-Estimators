#!/usr/bin/awk -f

#Removing lines with only one word

match($0, /^([[:alpha:]]{1,})$/){
	str=substr($0, RSTART, RLENGTH)
	sub(str, "", $0)
}1
