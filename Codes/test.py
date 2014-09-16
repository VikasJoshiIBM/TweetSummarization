#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Vikas
#
# Created:     12/04/2014
# Copyright:   (c) Vikas 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

f=open('temp/temp.txt','w')
x=[1.0,1.2,2,3,4]
f.write("\n".join(str(x)))
f.close()