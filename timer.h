#include <unistd.h>
#include <sys/time.h>
#include <stdio.h>

// Returns the current time of the day
double get_time()
{
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    return (double)timecheck.tv_sec + (double)timecheck.tv_usec*1e-6;
}

// Returns the elapsed time between the start time of the day and end time of the day
// Does not ignore OS operations
// Also measures any threads that interrupt (or idle time) in a parallel setting
double get_seconds(double start, double end)
{
    return end - start;
}


// Return the rate of data movement
// i.e. the rate at with data is read from memory
double get_rate(double seconds, long bytes)
{
    return bytes / seconds;
}

// Transform the rate to Gigabytes/second (more legible when printing)
double get_grate(double rate)
{
    return rate * 1e-9;
}
