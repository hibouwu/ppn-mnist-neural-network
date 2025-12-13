CC=gcc
CLFAGS=-O1

all:
	$(CC) $(CFLAGS) -c mnist_reader.c
	$(CC) $(CFLAGS) -c example_use.c
	$(CC) $(CFLAGS) -o example_use mnist_reader.o example_use.c

clean:
	rm -f *.o example_use
