CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
#CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS = -lm -pthread -g -g3 -gdwarf-2 -O3 -march=native -Wall -funroll-loops -Wno-unused-result
#CFLAGS = -lm -pthread -g -g3 -gdwarf-2 -O0 -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vecT

word2vecT : clean word2vecT.c vocab.c io.c
	$(CC) word2vecT.c vocab.c io.c -o word2vecT $(CFLAGS)

clean:
	rm -rf word2vecT
