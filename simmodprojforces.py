import numpy as np
import numba as nmb
from ctypes import c_float, c_int32, cast, byref, POINTER


def invsqrt(number):
    threehalfs = 1.5
    x2 = number * 0.5
    y = c_float(number) 

    i = cast(byref(y), POINTER(c_int32)).contents.value
    i = c_int32(0x5f3759df - (i >> 1))
    y = cast(byref(i), POINTER(c_float)).contents.value

    y = y * (1.5 - (x2 * y * y))
    return y


@nmb.jit(nopython=True)
def acc(m, G, pos1, pos2, hat):
    rsq = (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
    a = G * m / rsq
    return np.array([a * hat[0], a*hat[1]])


#@nmb.jit(nopython=True) # THis can be change to just calculating theta and stuff
def hat(pos1, pos2):
    x = pos1[0] - pos2[0]
    y = pos1[1] - pos2[1]
    inv = invsqrt(x ** 2 + y ** 2)
    return np.array([x * inv, y * inv])

