class Node:
    def __init__(self, data, depth, feature=None, theta=None):
        self.data = data
        self.depth = depth
        self.feature = feature
        self.theta = theta
        self.l = None
        self.r = None
        self.d_class = self.majority_class(data)

    def majority_class(self, data):
        if self.theta != None:
            return None
        pos = [d.y for d in data if d.y == 1]
        neg = [d.y for d in data if d.y == -1]
        return max((1, len(pos)), (-1, len(neg)), key=lambda i: i[1])[0]

    def print_tree(self):
        print('Feature: {}, Theta: {}, Depth: {}, Class: {}'.format(self.feature, self.theta, self.depth, self.d_class))
        print("data: {}".format([d.y for d in self.data]))
        if self.l: self.l.print_tree()
        if self.r: self.r.print_tree()