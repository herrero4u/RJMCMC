# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import statistics

import matplotlib.pyplot as plt
import numpy as np
import math
from random import random
from scipy.optimize import curve_fit


class Model:
    """Store model unknowns (nucleus number and coordinates)
    Attributes:
        x (float list): x coordinate of each nucleus
        y (float list): y coordinate of each nucleus
        npa (int): number of partitions
    """

    def __init__(self):
        """Model class constructor
        """
        self.x = []
        self.y = []
        self.npa = int

    def build_initial_model(self, x_min, x_max, y_dobs):
        npa_max = 10  # max number of partitions
        npa_min = 1  # min number of partitions
        self.npa = np.random.randint(npa_min, npa_max + 1)  # number of partitions
        print('npa', self.npa)
        for i in range(self.npa):
            self.x.append(np.random.uniform(x_min, x_max))  # x prior distribution
            self.y.append(np.random.uniform(min(y_dobs), max(y_dobs)))  # y prior distribution

    def build_proposed_model(self, current_model_):
        self.npa = current_model_.npa  # number of partitions
        # If u < 0.33, move, fonction move
        u = random()
        if u < 0.33:
            for k in range(self.npa):
                self.x.append(np.random.normal(current_model_.x[k], 0.7))  # x gaussian perturbation
                self.y.append(np.random.normal(current_model_.y[k], 8))  # y gaussian perturbation
        elif 0.33 <= u <= 0.66:
            for k in range(self.npa):
                self.x.append(current_model_.x[k])
                self.y.append(current_model_.y[k])
            move_index = self.x.index(np.random.choice(self.x)) # index of the random point to move
            self.x[move_index] = np.random.normal(self.x[move_index], 2)  # x gaussian perturbation
            self.y[move_index] = np.random.normal(self.y[move_index], 20)  # y gaussian perturbation
        else:
            for k in range(self.npa):
                self.x.append(current_model_.x[k])
                self.y.append(current_model_.y[k])
            move_index = self.x.index(np.random.choice(self.x))  # index of the random point to move
            self.x[move_index] = np.random.normal(self.x[move_index], 4)  # x gaussian perturbation


def compute_likelihood(x_dobs, y_dobs, x_i, y_i):
    likelihood = 0
    for j in range(len(x_dobs)):
        distance = []
        for nucleus in range(len(x_i)):
            distance.append(abs(x_dobs[j] - x_i[nucleus]))
            # print('distance', distance)
        index_min_dist = distance.index(min(distance))  # index of distance with the closest model point
        # print('distance min index', index_min_dist)
        likelihood += pow((y_dobs[j] - y_i[index_min_dist]) / sigma, 2)
    return likelihood


def draw_fit_curve(initial_model, mean_x, mean_y):
    xFit = np.arange(min(mean_x), max(mean_x), 0.01)
    if initial_model.npa >= 10:
        popt, _ = curve_fit(fifth_polynomial_regression, mean_x, mean_y)
        a, b, c, d, e, f = popt
        plt.plot(xFit, fifth_polynomial_regression(xFit, *popt), 'purple', label='fit param a=%5')
    elif 3 < initial_model.npa < 10:
        popt, _ = curve_fit(second_polynomial_regression, mean_x, mean_y)
        a, b, c = popt
        plt.plot(xFit, second_polynomial_regression(xFit, *popt), 'purple', label='fit param a=%5')
    elif 1 < initial_model.npa < 4:
        popt, _ = curve_fit(linear_regression, mean_x, mean_y)
        a, b = popt
        plt.plot(xFit, linear_regression(xFit, *popt), 'purple', label='fit param a=%5')


# define the true objective function
def fifth_polynomial_regression(x, a, b, c, d, e, f):
    return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + f


def second_polynomial_regression(x, a, b, c):
    return a * x + b * x ** 2 + c


def linear_regression(x, a, b):
    return a * x + b


fig, ax = plt.subplots()

# Plot true model
h1 = ax.hlines(y=0, xmin=0, xmax=1, linewidth=2, color='r', label='true model (np = 9)')
v1 = ax.vlines(x=1, ymin=0, ymax=20, linewidth=2, color='r')
h2 = ax.hlines(y=20, xmin=1, xmax=2.3, linewidth=2, color='r')
v2 = ax.vlines(x=2.3, ymin=0, ymax=20, linewidth=2, color='r')
h3 = ax.hlines(y=0, xmin=2.3, xmax=2.5, linewidth=2, color='r')
v3 = ax.vlines(x=2.5, ymin=-5, ymax=0, linewidth=2, color='r')
h4 = ax.hlines(y=-5, xmin=2.5, xmax=4, linewidth=2, color='r')
v4 = ax.vlines(x=4, ymin=-10, ymax=-5, linewidth=2, color='r')
h5 = ax.hlines(y=-10, xmin=4, xmax=6, linewidth=2, color='r')
v5 = ax.vlines(x=6, ymin=-20, ymax=-10, linewidth=2, color='r')
h6 = ax.hlines(y=-20, xmin=6, xmax=6.5, linewidth=2, color='r')
v6 = ax.vlines(x=6.5, ymin=-20, ymax=30, linewidth=2, color='r')
h7 = ax.hlines(y=30, xmin=6.5, xmax=8, linewidth=2, color='r')
v7 = ax.vlines(x=8, ymin=0, ymax=30, linewidth=2, color='r')
h8 = ax.hlines(y=0, xmin=8, xmax=9, linewidth=2, color='r')
v8 = ax.vlines(x=9, ymin=0, ymax=15, linewidth=2, color='r')
h9 = ax.hlines(y=15, xmin=9, xmax=10, linewidth=2, color='r')

x_dobs = []  # x coordinates of observed data
y_dobs = []  # y coordinates of observed data
sum_x = 0.0
step = 0.1  # observed data sampling on x
prev_i = 0
true_x = np.array([10, 23, 25, 40, 60, 65, 80, 90, 100])
true_y = np.array([0, 20, 0, -5, -10, -20, 30, 0, 15])
sigma = 10  # standard deviation
x_min = 0
x_max = 10
y_min = -60
y_max = 60

# Noisy points (observed data) creation
for i in range(len(true_x)):
    for j in range(prev_i, true_x[i]):
        sum_x += step
        x_dobs.append(sum_x)
        s = np.random.normal(true_y[i], sigma)  # add gaussian noise on points
        y_dobs.append(s)
    prev_i = true_x[i]

######### RJMCMC implementation ###########
# Build initial model
initial_model = Model()
initial_model.build_initial_model(x_min, x_max, y_dobs)
print(initial_model.x, initial_model.y, initial_model.npa)

# Compute likelihood for the initial model
current_likelihood = compute_likelihood(x_dobs, y_dobs, initial_model.x, initial_model.y)
# print('first likelihood', current_likelihood)
# print('initial model', initial_model)

# Set RJMCMC variables
current_model = initial_model
burn_in = 10000  # length of the burn-in period
nsamples = 50000  # total number of samples
accepted_models = 0  # number of accepted models
rejected_models = 0  # number of rejected models
model_space = []  # model space we want to sample

for sample in range(nsamples):
    u = random()

    # utiliser un bool de 1/3 pour choisir l action a realiser ?

    # Build proposed model with a perturbation from current model
    proposed_model = Model()
    proposed_model.build_proposed_model(current_model)

    # Compute likelihood of the proposed model
    proposed_likelihood = compute_likelihood(x_dobs, y_dobs, proposed_model.x, proposed_model.y)
    # print('proposed likelihood', proposed_likelihood)

    # Compute prior of the proposed model (i.e. check if within bounds)
    prior = 1
    for i in range(len(proposed_model.x)):
        if proposed_model.x[i] > x_max or proposed_model.x[i] < x_min or \
                proposed_model.y[i] > y_max or proposed_model.y[i] < y_min:
            prior = 0
    # print("prior", prior)

    alpha = prior * proposed_likelihood / current_likelihood  # acceptance term
    # print('alpha', alpha)
    if alpha > abs(math.log(u)):  # if accepted
        current_model = proposed_model
        current_likelihood = proposed_likelihood
        accepted_models += 1
    else:  # if rejected
        rejected_models += 1

    # Collect models in the chain if burn-in period has passed
    if sample >= burn_in:
        model_space.append(current_model)

print("accepted models", accepted_models)
print("rejected models", rejected_models)
acceptance_rate = 100 * accepted_models / nsamples
print("acceptance rate", acceptance_rate)

# Take the mean model from model space
mean_x = []
mean_y = []
for point in range(len(initial_model.x)):
    x = []
    y = []
    for model in model_space:
        # print("x", model.x[point])
        # print("y", model.y[point])
        x.append(model.x[point])
        y.append(model.y[point])
    mean_x.append(statistics.mean(x))
    mean_y.append(statistics.mean(y))
print("mean x", mean_x)
print("mean y", mean_y)
draw_fit_curve(initial_model, mean_x, mean_y)

# Figure plot
plt.scatter(x_dobs, y_dobs, c='green', label='observed data')
plt.scatter(initial_model.x, initial_model.y, c='blue', label='initial model')
plt.scatter(current_model.x, current_model.y, c='orange', label='last model')
plt.scatter(mean_x, mean_y, c='purple', label='mean model')
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend(loc='lower right')
plt.grid()
plt.show()
