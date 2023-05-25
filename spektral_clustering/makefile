CFlags = -ansi -Wall -Wextra -Werror -pedantic-errors -lm
spkmeans: spkmeans.o
	gcc -o spkmeans spkmeans.o $(CFlags)

spkmeans.o: spkmeans.c
	gcc -c spkmeans.c $(CFlags)
	
