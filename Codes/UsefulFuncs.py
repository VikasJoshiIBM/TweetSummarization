#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Vikas
#
# Created:     11/04/2014
# Copyright:   (c) Vikas 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def trunc(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    slen = len('%.*f' % (n, f))
    return str(f)[:slen]

def feq(a,b):
    if abs(a-b)<0.00001:
        return 1
    else:
        return 0
