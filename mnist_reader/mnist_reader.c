#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdbool.h>

// Define constants

#define IMAGE_MAGIC_NUMBER 2051
#define LABEL_MAGIC_NUMBER 2049
#define EXPECTED_ROWS 28
#define EXPECTED_COLS 28


// Helper function to check for big/little endianness
// (returns 0 for big endian, 1 for little endian)

bool is_little_endian() {
	int32_t a = 0x01234567;
	return (*((uint8_t*)(&a))) == 0x67;
}

// Helper function to invert the endianness of integers
// (probably needed since numbers are encoded in big endian in the MNIST files)

uint32_t reverseEndianness(int32_t i) {

	uint8_t c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int32_t)c1 << 24) + ((int32_t)c2 << 16) + ((int32_t)c3 << 8) + c4;
}



// Function to read n images at offset N from an MNIST image file pointer
// Returns the images in a uint8_t* (a 1D array of size n*EXPECTED_ROW_COL*EXPECTED_ROW_COL)
// In this array, images are stored one after another in row-major ordering.
// If something goes wrong while reading the file, the function returns NULL
// NOTE You will probably need to rescale the range of the values contained
// in the output array from [0, 255] to [0, 1] and use floating point numbers

uint8_t* readMnistImages(
	FILE* imageFile,
	int32_t N, int32_t n
) {

	// Infos about the file format of MNIST files can be found at :
	// http://yann.lecun.com/exdb/mnist/
	// (see last two sections of the page)

	if(n <= 0) {
		fprintf(stderr, "ERROR: You must request at least 1 image\n");
		return NULL;
	}

	if(N < 0) {
		fprintf(stderr, "ERROR: You must specify an offset >= 0\n");
		return NULL;
	}


	// Initialize variables into which file constants will be read
	int32_t magicNumber, nImages, nRows, nCols;


	// Read and verify all file constants
	fread(&magicNumber, sizeof(int32_t), 1, imageFile);
	if(is_little_endian()) {
		magicNumber = reverseEndianness(magicNumber);
	}
	if(magicNumber != IMAGE_MAGIC_NUMBER) {
		fprintf(stderr, "ERROR: image file seems invalid (wrong magic number)\n");
		return NULL;
	}

	fread(&nImages, sizeof(int32_t), 1, imageFile);
	if(is_little_endian()) {
		nImages = reverseEndianness(nImages);
	}
	if(nImages < n) {
		fprintf(stderr, "ERROR: you requested %d images but the file only contains %d\n", n, nImages);
		return NULL;
	} else if(nImages < N + n) {
		fprintf(stderr, "ERROR: reading %d images from offset %d would exceed file size (%d images)\n", n, N, nImages);
		return NULL;
	}

	fread(&nRows, sizeof(int32_t), 1, imageFile);
	if(is_little_endian()) {
		nRows = reverseEndianness(nRows);
	}
	if(nRows != EXPECTED_ROWS) {
		fprintf(stderr, "ERROR: files contains images of %d rows, expected %d\n", nRows, EXPECTED_ROWS);
		return NULL;
	}

	fread(&nCols, sizeof(int32_t), 1, imageFile);
	if(is_little_endian()) {
		nCols = reverseEndianness(nCols);
	}
	if(nCols != EXPECTED_COLS) {
		fprintf(stderr, "ERROR: files contains images of %d cols, expected %d\n", nCols, EXPECTED_COLS);
		return NULL;
	}


	// Allocate array into which images will be stored
	uint8_t * images = malloc(n*EXPECTED_ROWS*EXPECTED_COLS*sizeof(uint8_t));


	// Move to the specified offset
	fseek(imageFile, N*EXPECTED_ROWS*EXPECTED_COLS*sizeof(int8_t), SEEK_CUR);

	// Read file data into the "images" array
	fread(images, sizeof(int8_t), n*EXPECTED_ROWS*EXPECTED_COLS, imageFile);

	return images;
}



// Function to read n labels at offset N from an MNIST labels file pointer
// Returns the labels in a uint8_t* (a 1D array of size n*EXPECTED_ROW_COL*EXPECTED_ROW_COL)
// If something goes wrong while reading the file, the function returns NULL

uint8_t* readMnistLabels(
	FILE* labelFile,
	int32_t N, int32_t n
) {

	// Infos about the file format of MNIST files can be found at :
	// http://yann.lecun.com/exdb/mnist/
	// (see last two sections of the page)

	if(n <= 0) {
		fprintf(stderr, "ERROR: You must request at least 1 label\n");
		return NULL;
	}

	if(N < 0) {
		fprintf(stderr, "ERROR: You must specify an offset >= 0\n");
		return NULL;
	}


	// Initialize variables into which file constants will be read
	int32_t magicNumber, nLabels;


	// Read and verify all file constants
	fread(&magicNumber, sizeof(int32_t), 1, labelFile);
	if(is_little_endian()) {
		magicNumber = reverseEndianness(magicNumber);
	}
	if(magicNumber != LABEL_MAGIC_NUMBER) {
		fprintf(stderr, "ERROR: label file seems invalid (wrong magic number)\n");
		return NULL;
	}

	fread(&nLabels, sizeof(int32_t), 1, labelFile);
	if(is_little_endian()) {
		nLabels = reverseEndianness(nLabels);
	}
	if(nLabels < n) {
		fprintf(stderr, "ERROR: you requested %d labels but the file only contains %d\n", n, nLabels);
		return NULL;
	} else if(nLabels < N + n) {
		fprintf(stderr, "ERROR: reading %d labels from offset %d would exceed file size (%d labels)\n", n, N, nLabels);
		return NULL;
	}


	// Allocate array into which labels will be stored
	uint8_t * labels = malloc(n*sizeof(uint8_t));


	// Move to the specified offset
	fseek(labelFile, N*sizeof(int8_t), SEEK_CUR);

	// Read file data into the "labels" array
	fread(labels, sizeof(int8_t), n, labelFile);

	return labels;
}



// Function to display a digit in ASCII given a pointer to its first element
// (when digit is in row-major ordering, and elements are of type uint8t)
// This can be used to check that you are feeding your neural network correct data,
// or that your data and labels match correctly, for instance.
void printAsciiDigit(uint8_t* digit) {

	printf("┌");
	for(unsigned i=0; i<EXPECTED_COLS; i++) {
		printf("─");
	}
	printf("┐\n");

	for(unsigned i=0; i<EXPECTED_ROWS; i++) {
		printf("│");

		for(unsigned j=0; j<EXPECTED_COLS; j++) {

			if(digit[i * EXPECTED_ROWS + j] == 0) {
				printf(" ");
			} else if(digit[i * EXPECTED_ROWS + j] < 64) {
				printf("░");
			} else if(digit[i * EXPECTED_ROWS + j] < 128) {
				printf("▒");
			} else if(digit[i * EXPECTED_ROWS + j] < 192) {
				printf("▓");
			} else {
				printf("█");
			}

		}

		printf("│\n");
	}

	printf("└");
	for(unsigned i=0; i<EXPECTED_COLS; i++) {
		printf("─");
	}
	printf("┘\n");

}
