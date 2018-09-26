#!/bin/bash

text_cleaner(){
	./date_remove.awk $1 		|
	./long_number_remove.awk 	|
	./headline_number_remove.awk	|
	./uppercase_remove.awk		|
	./whitespace_remove.awk		|
	./numeric_line_remove.awk	|
	./singleton_remove.awk		|
	./blank_line_remove.awk		

}

text_cleaner $1 > "temp1"

slicing(){
	./sentence_cutter.x temp1 
}

slicing temp1 > "temp2"

rm temp1


text_refine(){
	./headline_number_remove.awk temp2	|
	./whitespace_remove.awk			|
	./singleton_remove.awk			|
	./blank_line_remove.awk		
}

text_refine temp2 > "output"
rm temp2

exploration(){
	
	./descriptive_stats.x output
} 

exploration output  > "stats"


