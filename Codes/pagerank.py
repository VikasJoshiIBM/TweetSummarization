#generic page rank implementation

import operator
import os, re
from scipy import sparse
from numpy import matrix,ones,zeros

PAGE_RANK_ALPHA = .1

def transpose(links):
    ret = dict([(key,{}) for key in links])
    for dst in links:
        for src in links[dst]:
            ret[src][dst] = links[dst][src]
    return ret

class PageRank(object):
    def __init__(self,links):
        #list of keys
        self._links = links
        self._keys = links.keys()
        self._key_index = dict(zip(self._keys,xrange(len(self._keys))))

    def calculate(self):
        size = len(self._keys)
        teleport = [PAGE_RANK_ALPHA if len(self._links[key]) else 1.0 for key in self._keys]
        teleport = matrix(teleport).getT()/size
        trans_prob = sparse.lil_matrix((size,size))

        for src in self._keys:
            links = self._links[src]
            if len(links):
                row = self._key_index[src]
                row_val = (1-PAGE_RANK_ALPHA)/sum(links.values())
                for dest in links:
                    col = self._key_index[dest]
                    trans_prob[row,col]=row_val*links[dest]
        print "built matrix for page rank"

        trans_prob = sparse.csc_matrix(trans_prob)
        old = matrix(ones(size))/size
        for iteration in xrange(9):
            print iteration
            teleport_prob = (old*teleport)[0,0]
            new = teleport_prob+old*trans_prob
            old = new
        return dict(zip(self._keys, new.flat))

def main(args):
    pr = PageRank({3:{5:9,7:1}, 5:{7:1}, 7:{}})
    results = pr.calculate()
    print results


# this little helper will call main() if this file is executed from the command
# line but not call main() if this file is included as a module
if __name__ == "__main__":
    import sys
    main(sys.argv)