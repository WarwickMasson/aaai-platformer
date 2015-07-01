'''
This file contains useful utility functions.
'''
import numpy as np

def vector(xvalue, yvalue):
    ''' Returns a 2D numpy vector. '''
    return np.array([float(xvalue), float(yvalue)])

def vector_to_tuple(vect):
    ''' Converts a 2D vector to a tuple. '''
    return (vect[0], vect[1])

def to_matrix(vect):
    ''' Turns a vector into a single column matrix. '''
    return np.array([vect]).T
