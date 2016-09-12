#!/usr/bin/python3
import numpy

if __name__ == "__main__":
    raw_data = numpy.loadtxt(open("Concrete_Data.csv", "rb"), delimiter=",", skiprows=1)
    v = 1
