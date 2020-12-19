#include <stdio.h>
#include "book.h"

int main(void)
{
    int whichDevice;
    cudaDeviceProp prop;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap)
    {
        printf("Device do not supports device overlap\n");
    }
    else
    {
        printf("Passed\n");
    }
}

