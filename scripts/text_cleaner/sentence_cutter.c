/* Sentence cutter. If longer than n words (" " separated)
cut the sentence with newline */

#include<stdio.h>
#include<string.h>
#include<ctype.h>

int mean(int, int);

/* MAIN */

int main(int argc, char *argv[]){

	FILE *inp = fopen(argv[1], "r");
	char c;
	int word_counter = 0;

	while((c = fgetc(inp)) != EOF){	
		
		printf("%c", c);
		
		if(isspace(c))
			++word_counter;
		/* Cutter */	
		if(c == '\n')
			word_counter = 0;

		if(word_counter == 16){
			printf("\n");
			word_counter = 0;
		}
	}

	return 0;
}


