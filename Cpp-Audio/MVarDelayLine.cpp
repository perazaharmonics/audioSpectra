#include <iostream>
#include <cmath>
#include <algorithm>

#include "FileWvIn.h"
#include "FileWvOut.h"
#include "FileLoop.h"
#include "Stk.h"

using namespace stk;
const int N = 1024; // Buffer size, assuming a value. Please change it accordingly.
static double A[N];
static double g = 1; // fixed delay line
static double* rptr = A; // READ PTR
static double* wptr = A; // WRITE PTR

/*
 * Function: setdelay
 * --------------------
 * Adjusts the read pointer to set the delay for the delay line.
 *
 * M: The number of samples to delay.
 *
 * returns: The position of the read pointer after being adjusted.
 */
double setdelay(int M) {

    rptr = wptr - M;
    while (rptr < A) {
        rptr += N;
    }
    return (rptr - A); // It should return a value, I'm returning the position as an example.
}

/*
 * Function: delayline
 * --------------------
 * Implements a delay line with linear interpolation for fractional delay.
 *
 * x: The input sample.
 *
 * returns: The delayed sample.
 */
static int windex = 0; // Write index
static int rindex = 0; // Read index

double delayline(double x)
{
    double y;
    A[windex] = x;

    double frac = rindex + g - std::floor(rindex + g);  // Calculate the fractional part for interpolation
    int rpi = static_cast<int>(std::floor(rindex + g)) % N;  // Read index with gain considered and wrapped around

    y = (1 - frac) * A[rpi] + frac * A[(rpi + 1) % N];  // Linear interpolation

    windex = (windex + 1) % N; // Increment and wrap the write index
    rindex = (rindex + 1) % N; // Increment and wrap the read index

    return y;
}


