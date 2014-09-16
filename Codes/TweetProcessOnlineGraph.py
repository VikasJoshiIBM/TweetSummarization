#from numpy import *
import math
import numpy
#import nltk
import sys
import json
from pprint import pprint

values = []
with open('data.json') as json_file:
	for line in json_file:
		values.append( json.loads(line) )

