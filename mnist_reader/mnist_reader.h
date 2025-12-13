#pragma once

#include <inttypes.h>
#include <stdbool.h>


uint8_t* readMnistImages(
	FILE* imageFile,
	int32_t N, int32_t n
);
uint8_t* readMnistLabels(
	FILE* labelFile,
	int32_t N, int32_t n
);


void printAsciiDigit(uint8_t* digit);
