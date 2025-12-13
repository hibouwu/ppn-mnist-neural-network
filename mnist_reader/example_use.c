#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "mnist_reader.h"

int main(void) {
	FILE* imageFile = fopen("mnist/train-images-idx3-ubyte", "r");
	FILE* labelFile = fopen("mnist/train-labels-idx1-ubyte", "r");

	if(imageFile == NULL || labelFile == NULL) {
		fprintf(stderr, "ERROR: At least one file could not be read.\n");
		return 1;
	}

	// Read 10 images from the 50th image
	uint8_t* images = readMnistImages(imageFile, 50, 10);
	uint8_t* labels = readMnistLabels(labelFile, 50, 10);

	fclose(imageFile);
	fclose(labelFile);

	for(int i=0; i<10; i++) {
		printf("i=%d\n", i);
		printAsciiDigit(images + i*784*sizeof(uint8_t));
		printf("Label=%d\n", *(labels+i));
		printf("********************************\n");
	}

	free(images);
	free(labels);

	return 0;
}
