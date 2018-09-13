/* Basic stats */

#include<stdio.h>
#include<string.h>
#include<ctype.h>

int mean(int, int);

/* MAIN */

int main(int argc, char *argv[]){

	FILE *inp = fopen(argv[1], "r");
	char c;
	int line_counter = 0;
	int word_counter = 0;
	int char_counter = 0;

	while((c = fgetc(inp)) != EOF){			/* Counters */
		++char_counter;
		
		if(c == '\n')
			++line_counter;
		
		if(isspace(c))
			++word_counter;
	}

	printf("Lines:\t\t\t\t%d\n"			/* Print counters */
		"Words:\t\t\t\t%d\n",
		line_counter, word_counter);
	
	printf("Mean words/line:\t\t%d\n", mean(word_counter, line_counter));
	
	printf("Mean chars/line:\t\t%d\n", mean(char_counter - line_counter, 
		line_counter));
	
	return 0;
}

/* AUX */

int mean(int n1, int n2){
	return n1 / n2;
}







