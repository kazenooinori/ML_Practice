import sys
import numpy as np

class PLA:
    def __init__(self, dim):
        # learning rate
        self.eta = 1
        # init with all weights set to 0
        self.weights = np.zeros(dim)
        # keep track of update round
        self.updates = 0

    def cal_new_y(self, xs):
        # if weights are 0s, return -1 as default
        if not np.any(self.weights):
            return -1
        else:
            return np.sum([w*x for w, x in zip(self.weights, xs)])

    def check_same_sign(self, y, new_y):
        return ((y < 0) == (new_y < 0))

    def run(self, feats, label):
        passed = 0
        # stop only when there is no mistake
        while passed < len(label):
            for xs, y in zip(feats, label):
                # update weight when there is a mistake
                if not self.check_same_sign(int(y), self.cal_new_y(xs)):
                    new_weights = [w+y*x*self.eta for w, x in zip(self.weights, xs)]
                    #print '[update ' + str(self.updates) + '] passed:' + str(passed) + '[new weight]' + str(new_weights)
                    self.weights = new_weights
                    self.updates += 1
                    passed = 0
                    break
                else:
                    passed += 1

    def print_stats(self):
        print 'update rounds:' + str(self.updates)
        print 'weights:' + str(self.weights)

if __name__ == '__main__':
    # load data
    flines = open(sys.argv[1], 'r').readlines()
    feats = np.array([ line.strip().split('\t')[0].split(' ') for line in flines]).astype(float)
    labels = np.array([ line.strip().split('\t')[1] for line in flines]).astype(int)

    # append x0 = 1
    feats = np.column_stack((np.ones(len(feats)),feats))
    (countOfData, featsDim) = feats.shape

    # run pla for data
    pla = PLA(featsDim)
    pla.run(feats, labels)
    pla.print_stats()
