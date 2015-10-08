import numpy as np

class P_BASE:
    def __init__(self, dim):
        # learning rate
        self.eta = 1
        # init with all weights set to 0
        self.weights = np.zeros(dim)
        # keep track of update round
        self.updates = 0
        # amount of error
        self.err = 0

    def cal_new_y(self, xs):
        # if weights are 0s, return -1 as default
        if not np.any(self.weights):
            return -1
        else:
            return np.sum([w*x for w, x in zip(self.weights, xs)])

    def check_same_sign(self, y, new_y):
        return ((y < 0) == (new_y < 0))

    def update_weights(self, y, xs):
        new_weights = [w+y*x*self.eta for w, x in zip(self.weights, xs)]
        #print '[update ' + str(self.updates) + '] passed:' + str(passed) + '[new weight]' + str(new_weights)
        return new_weights

class PLA(P_BASE):
    def learn(self, feats, labels):
        passed = 0
        # stop only when there is no mistake
        while passed < len(labels):
            for xs, y in zip(feats, labels):
                # update weight when there is a mistake
                if not self.check_same_sign(int(y), self.cal_new_y(xs)):
                    self.weights = self.update_weights(y, xs)
                    self.updates += 1
                    passed = 0
                    break
                else:
                    passed += 1

    def print_stats(self):
        print 'update rounds:' + str(self.updates)
        print 'weights:' + str(self.weights)

class Pocket(P_BASE):
    def __init__(self, dim, iterBound):
        P_BASE.__init__(self, dim)
        self.iter_bound = iterBound

    def learn(self, feats, labels):

        pocket_weights = np.zeros(featsDim)
        pocket_err = 0

        for run_rounds in xrange(self.iter_bound):
            # count the number of errors
            for data in self.find_err_data(feats, labels):
                self.err += 1
                if self.err == 1:
                    new_weights = self.update_weights(data[0], data[1])

            # compare the amount of errors, if the new
            #print 'self.err:' + str(self.err)
            if self.err == 0:
                break
            elif run_rounds == 0 or pocket_err > self.err:
                pocket_err = self.err
                pocket_weights = self.weights
                #print '[round ' + str(run_rounds) + '] err: ' + str(pocket_err) + ', pocket weights:' + str(pocket_weights)

            # prepare weight for next round
            self.err = 0
            self.weights =  new_weights

        self.weights = pocket_weights
        self.err = pocket_err

    def find_err_data(self, feats, labels):
        for xs, y in zip(feats, labels):
            if not self.check_same_sign(int(y), self.cal_new_y(xs)):
                yield [y, xs]

    def test(self, feats, labels):
        err = 0
        for data in self.find_err_data(feats, labels):
            err += 1
        self.errRate = float(err)/len(labels)

    def print_stats(self):
        print 'iteration bound: ' + str(self.iter_bound)
        print 'error num: ' + str(self.err)
        if hasattr(self, 'errRate'):
            print 'error rate: ' + str(self.errRate)
        print 'weights: ' + str(self.weights)

if __name__ == '__main__':
    import sys, random

    def prepare_data(fileName):
        # load data
        flines = open(fileName, 'r').readlines()
        feats = np.array([ line.strip().split('\t')[0].split(' ') for line in flines]).astype(float)
        labels = np.array([ line.strip().split('\t')[1] for line in flines]).astype(int)

        # append x0 = 1
        feats = np.column_stack((np.ones(len(feats)),feats))
        return feats, labels

    def random_test(times, func, feats, labels):
        for time in xrange(times):
            data = zip(feats, labels)
            random.shuffle(data)
            s_feats, s_labels = zip(*data)
            func(s_feats, s_labels)

    # [training phase]
    train_data_feats, train_data_labels = prepare_data(sys.argv[1])
    (countOfData, featsDim) = train_data_feats.shape

    # run pla for data
    #pla = PLA(featsDim)
    #pla.learn(feats, labels)
    #pla.print_stats()
    pt = Pocket(featsDim, 50)
    pt.learn(train_data_feats, train_data_labels)

    # [testing phase]
    # load testing data
    test_data_feats, test_data_labels = prepare_data(sys.argv[2])
    pt.test(test_data_feats, test_data_labels)
    pt.print_stats()
