import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use("ggplot")


class Support_Vector_Machine():
    def __init__(self, visualization=True):
        self.data = None
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)



    # train
    def fit(self, data):
        # This is optimization problem # convex Optimization problem
        self.data = data
        opt_dict = {}

        # {||W||:[w,b] = sqrt"W2 + b2"}
        transforms = [[1, -1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        all_data = []

        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.max_feature_value = max(all_data)
        all_data = None

        # Support vectors Yi(Xi.W+b) = 1, how close to ONE(1) ?
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense
                      self.max_feature_value * 0.001,
                      self.max_feature_value * 0.0001]

        # extremely expensive
        b_range_multiple = 5
        # we don't need to take as small of steps
        # with b as we do w
        b_multiple = 5

        latest_optimum = self.max_feature_value * 10

        for step_in in step_sizes:
            w = np.array([latest_optimum, latest_optimum])  # This is max value
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step_in * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b)>=1
                        #
                        ## add a break here later
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(xi, w_t) + b) >= 1:
                                    found_option = False
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

            if w[0] < 0:
                optimized = True
                print("Optimized step...")
            else:
                w = w - step_in

            norms = sorted([n for n in opt_dict])
            # ||w||:[w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step_in * 2

    # test
    def predict(self, features):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0],features[1],s=200, marker="*", c=self.colors[classification])
        return classification

    def visualize(self):
        hper_planes_map = []
        for i in self.data:
            for x in self.data[i]:
                hper_planes_map.append(self.ax.scatter(x[0],x[1]))
