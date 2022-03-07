"""
This script recreates the regression example presented in "Transdimensional inference in the geosciences" paper
(Sambridge et al. 2013) using transdimensional inversions.
Through RJMCMC iterations, perturbations as moves, births and deaths of Voronoi nuclei are proposed to explore the
model space and extract a mean model that aims to fit the true model as fine as possible.
The y coordinate of model partitions is the parameter of interest in the Bayesian inversion. The model dimension k
concerns the number of model Voronoi nuclei.
"""

__author__ = "Julien Herrero"
__contact__ = "julien.herrero@univ-lorraine.fr"
__copyright__ = "RING Team"
__date__ = "2022-03-01"
__version__ = "1"

import math
import statistics
import numpy as np
import matplotlib.pyplot as plt
import shapely
from random import random
from scipy.optimize import curve_fit
from shapely.geometry import LineString, Point


class Model:
    """Store model unknowns (Voronoi nuclei number and coordinates) and parameterization
    Attributes:
        x (float list): x coordinate of each nucleus
        y (float list): y coordinate of each nucleus
        lines (float list): store x-y coordinates of model partition segments
        npa (int): number of partitions
        npa_min (int): minimal possible number of partitions
        npa_max (int): maximal possible number of partitions
        phi (float): misfit function which quantifies the agreement between simulated and observed data
        birth_param (tuple): store nucleus birth parameterization
        death_param (tuple): store nucleus death parameterization
        curr_perturbation (str): state of the perturbation at current iteration
    """

    def __init__(self):
        """Model class constructor
        """
        self.x = []
        self.y = []
        self.lines = []
        self.npa_min = int
        self.npa_max = int
        self.npa = int
        self.phi = float
        self.birth_param = tuple
        self.death_param = tuple
        self.curr_perturbation = str

    def build_initial_model(self, x_min, x_max, y_dobs):
        """Build the initial model of the chain from a uniform prior distribution
        :param int x_min: x minimal coordinate in the field
        :param int x_max: x maximal coordinate in the field
        :param int y_dobs: y coordinates of observed data used as y min-max coordinates
        """
        self.npa_min = 1  # min number of partitions
        self.npa_max = 50  # max number of partitions
        self.npa = np.random.randint(self.npa_min, self.npa_max + 1)  # number of partitions
        print("initial model npa", self.npa)
        for k in range(self.npa):
            self.x.append(np.random.uniform(x_min, x_max))  # x prior distribution
            self.y.append(np.random.uniform(min(y_dobs), max(y_dobs)))  # y prior distribution

    def build_proposed_model(self, current_model_):
        """Build a proposed model from the current model with a random perturbation (birth, death, and move)
        :param Model current_model_: model used in the chain at the current iteration
        """
        self.npa = current_model_.npa
        self.npa_min = current_model_.npa_min
        self.npa_max = current_model_.npa_max
        perturb_type = np.random.random_sample()  # random number to choose the perturbation type to apply
        if perturb_type < 0.33:
            self.curr_perturbation = "move"
            self.move(current_model_)
        elif 0.33 <= perturb_type <= 0.66:
            self.curr_perturbation = "birth"
            self.birth(current_model_)
        else:
            self.curr_perturbation = "death"
            self.death(current_model_)

    def move(self, current_model_):
        """Propose a random move of nuclei as a perturbation of the model"""
        move_type = np.random.random_sample()  # random number to choice the move perturbation to apply
        if move_type < 0.33:  # move all nuclei (gaussian perturbation)
            for k in range(self.npa):
                self.x.append(np.random.normal(current_model_.x[k], 0.3))
                self.y.append(np.random.normal(current_model_.y[k], 5))
        elif 0.33 <= move_type <= 0.66:  # move one nucleus in x & y axes (gaussian perturbation)
            for k in range(self.npa):
                self.x.append(current_model_.x[k])
                self.y.append(current_model_.y[k])
            move_index = self.x.index(np.random.choice(self.x))  # index of the random point to move
            self.x[move_index] = np.random.normal(self.x[move_index], 0.5)
            self.y[move_index] = np.random.normal(self.y[move_index], 10)
        else:  # move one nucleus in x-axis (gaussian perturbation)
            for k in range(self.npa):
                self.x.append(current_model_.x[k])
                self.y.append(current_model_.y[k])
            move_index = self.x.index(np.random.choice(self.x))  # index of the random point to move
            self.x[move_index] = np.random.normal(self.x[move_index], 1)

    def full_rand_birth(self, current_model_):
        """Propose a full random nucleus birth from uniform prior distribution"""
        self.npa += 1  # increasing number of partitions
        print('npa after birth', self.npa)
        for k in range(self.npa - 1):
            self.x.append(current_model_.x[k])
            self.y.append(current_model_.y[k])
        self.x.append(np.random.uniform(x_min, x_max))  # birth from x prior distribution
        self.y.append(np.random.uniform(min(y_dobs), max(y_dobs)))  # birth from y prior distribution

    def full_rand_death(self, current_model_):
        """Propose a full random nucleus death from uniform prior distribution"""
        if self.npa > 1:  # the model dimension can not be less than 1
            self.npa -= 1  # decreasing number of partitions
            print('npa after death', self.npa)
            for k in range(self.npa + 1):
                self.x.append(current_model_.x[k])
                self.y.append(current_model_.y[k])
            death_index = self.x.index(np.random.choice(self.x))  # index of the random point to delete
            del self.x[death_index]  # death from x prior distribution
            del self.y[death_index]  # death from y prior distribution
        else:
            print('npa = ', self.npa, 'so nothing is done')
            for k in range(self.npa):
                self.x.append(current_model_.x[k])
                self.y.append(current_model_.y[k])

    def birth(self, current_model_):
        """Propose a nucleus birth distributed from a gaussian law"""
        self.npa += 1  # increasing number of partitions
        sigma2 = 5
        print('npa after birth', self.npa)
        for k in range(self.npa - 1):
            self.x.append(current_model_.x[k])
            self.y.append(current_model_.y[k])
        self.x.append(np.random.uniform(x_min, x_max))  # x random birth
        distance = []
        for nucleus in range(self.npa - 1):
            distance.append(abs(self.x[self.npa - 1] - self.x[nucleus]))
        index_min_dist = distance.index(min(distance))
        vi = self.y[index_min_dist]  # current y value (closest nucleus from the birth)
        self.y.append(np.random.normal(vi, sigma2))  # y birth from gaussian proposal probability
        vnp1 = self.y[self.npa - 1]  # new y parameter value
        self.birth_param = (vnp1, vi)

    def death(self, current_model_):
        """Propose a nucleus death distributed from a gaussian law"""
        if self.npa > 1:
            self.npa -= 1  # decreasing number of partitions
            print('npa after death', self.npa)
            for k in range(self.npa + 1):
                self.x.append(current_model_.x[k])
                self.y.append(current_model_.y[k])
            death_index = self.x.index(np.random.choice(self.x))  # index of the random point to delete
            point_to_kill = self.x[death_index]  # value of the nucleus to be killed
            temp_x = []
            for nucleus in range(len(self.x)):
                temp_x.append(self.x[nucleus])
            temp_x[death_index] = 100000
            del self.x[death_index]
            print("temp", temp_x)
            distance = []
            print("self x just after death", self.x)
            print("self y just before death", self.y)
            print("point to kill", point_to_kill)
            for nucleus in range(len(self.y)):
                print(" range npa", range(self.npa))
                print("nucleus", nucleus)
                distance.append(abs(point_to_kill - temp_x[nucleus]))
            print("distance", distance)
            index_min_dist = distance.index(min(distance))  # index of distance with the closest model point
            print("index min", index_min_dist)
            vi = self.y[death_index]
            vj = self.y[index_min_dist]
            self.death_param = (vj, vi)
            print('vj1', vj)
            print('vi1', vi)
            print('death param1', self.death_param)
            del self.y[death_index]
        else:
            self.curr_perturbation = "nothing"
            print('npa = ', self.npa, 'so nothing is done')
            for k in range(self.npa):
                self.x.append(current_model_.x[k])
                self.y.append(current_model_.y[k])
        print("self x after death", self.x)
        print("self y after death", self.y)

    def compute_prior(self):
        prior = 1
        if self.npa < self.npa_min or self.npa > self.npa_max:
            prior = 0
        else:
            for nucleus in range(len(self.x)):
                if self.x[nucleus] > x_max or self.x[nucleus] < x_min or \
                        self.y[nucleus] > max(y_dobs) or self.y[nucleus] < min(y_dobs):
                    prior = 0
        # print("prior", prior)
        return prior

    def compute_likelihood(self):
        return math.exp(-(1 / 2) * self.phi)

    def compute_phi(self, x_dobs, y_dobs):
        self.phi = 0
        for j in range(len(x_dobs)):
            distance = []
            for nucleus in range(len(self.x)):
                distance.append(abs(x_dobs[j] - self.x[nucleus]))
                # print('distance', distance)
            index_min_dist = distance.index(min(distance))  # index of distance with the closest model point
            # print('distance min index', index_min_dist)
            self.phi += pow(y_dobs[j] - self.y[index_min_dist], 2) / pow(sigma, 2)

    def draw_lines(self):
        # color = list(np.random.uniform(range(0, 1), size=3))
        xy_coord = []
        for i in range(len(self.x)):
            xy_coord.append((self.x[i], self.y[i]))
            print("xy_coord", xy_coord)
        sorted_list = sorted(xy_coord, key=lambda xy_: xy_[0])  # sort by x
        print("sorted list", sorted_list)
        for i in range(len(self.x)):
            self.x[i] = sorted_list[i][0]
            self.y[i] = sorted_list[i][1]
        print("self x", self.x)
        print("self y", self.y)
        xmin_line = 0
        change_point = 0
        for i in range(1, len(self.x)):
            change_point = (self.x[i] + self.x[i - 1]) / 2
            plt.hlines(y=self.y[i - 1], xmin=xmin_line, xmax=change_point, linewidth=2, color='b')
            plt.vlines(x=change_point, ymin=self.y[i - 1], ymax=self.y[i], color='b')
            xmin_line = change_point
        plt.hlines(y=self.y[len(self.y) - 1], xmin=change_point, xmax=x_max)

    def store_lines(self):
        xy_coord = []
        for i in range(len(self.x)):
            xy_coord.append((self.x[i], self.y[i]))  # list of tuples gathering x and y nucleus coordinates
        sorted_list = sorted(xy_coord, key=lambda xy_: xy_[0])  # sort by x
        for i in range(len(self.x)):
            self.x[i] = sorted_list[i][0]
            self.y[i] = sorted_list[i][1]
        xmin_line = 0
        change_point = 0
        line = np.zeros((2, 2))
        for i in range(1, len(self.x)):
            change_point = (self.x[i] + self.x[i - 1]) / 2
            line[0][0] = xmin_line
            line[0][1] = self.y[i - 1]
            line[1][0] = change_point
            line[1][1] = line[0][1]
            print("line", line)
            self.lines.append(line)
            xmin_line = change_point
            line = np.zeros((2, 2))
        line[0][0] = change_point
        line[0][1] = self.y[len(self.y) - 1]
        line[1][0] = x_max
        line[1][1] = line[0][1]
        print("line", line)
        self.lines.append(line)
        print("self line", self.lines)


def compute_acceptance(current_model_, proposed_model_):
    # Compute prior of the proposed model (i.e. check if npa, x and y within bounds)
    prior = 1
    if proposed_model_.npa < initial_model.npa_min or proposed_model_.npa > initial_model.npa_max:
        prior = 0
    else:
        for i in range(len(proposed_model_.x)):
            if proposed_model_.x[i] > x_max or proposed_model_.x[i] < x_min or \
                    proposed_model_.y[i] > max(y_dobs) or proposed_model_.y[i] < min(y_dobs):
                prior = 0
    # print("prior", prior)
    print("perturbation", proposed_model_.curr_perturbation)
    sigma2 = 5  # standard deviation from Gaussian probability density
    delta_v = max(y_dobs) - min(y_dobs)  # uniform y distribution
    if proposed_model_.curr_perturbation == "move":
        # Compute likelihood of the proposed model
        proposed_likelihood = proposed_model_.compute_likelihood()
        current_likelihood = current_model_.compute_likelihood()
        print("proposed likelihood", proposed_likelihood)
        print("current likelihood", current_likelihood)
        print("ratio", proposed_likelihood / current_likelihood)
        # print('proposed likelihood', proposed_likelihood)
        return proposed_likelihood / current_likelihood  # acceptance term
    elif proposed_model_.curr_perturbation == "birth":
        vnp1 = proposed_model_.birth_param[0]
        vi = proposed_model_.birth_param[1]
        print("1st term", sigma2 * math.sqrt(2 * math.pi) / (max(y_dobs) - min(y_dobs)))
        print("proposal proba", (pow(vnp1 - vi, 2)))
        print("proposal ratio", (pow(vnp1 - vi, 2) / (2 * pow(sigma2, 2))))
        print("little test", (pow(vnp1 - vi, 2) / (2 * 4)))
        print("proposed phi", proposed_model_.phi)
        print("current phi", current_model_.phi)
        print("ratio phi ", ((proposed_model_.phi - current_model_.phi) / 2))
        print("2nd term", (pow(vnp1 - vi, 2) / (2 * pow(sigma2, 2))) - ((proposed_model_.phi - current_model_.phi) / 2))
        print("exp",
              math.exp((pow(vnp1 - vi, 2) / (2 * pow(sigma2, 2))) - ((proposed_model_.phi - current_model_.phi) / 2)))
        print("result", (sigma2 * math.sqrt(2 * math.pi) / (max(y_dobs) - min(y_dobs))) * math.exp(
            (pow(vnp1 - vi, 2) / (2 * pow(sigma2, 2))) - ((proposed_model_.phi - current_model_.phi) / 2)))
        return (sigma2 * math.sqrt(2 * math.pi) / delta_v) * math.exp(
            (pow(vnp1 - vi, 2) / (2 * pow(sigma2, 2))) - ((proposed_model_.phi - current_model_.phi) / 2))
    elif proposed_model_.curr_perturbation == "death":
        vj = proposed_model_.death_param[0]
        vi = proposed_model_.death_param[1]
        print('vj', proposed_model_.death_param[0])
        print('vi', proposed_model_.death_param[1])
        print('death param', proposed_model_.death_param)
        print("1st term", (max(y_dobs) - min(y_dobs)) / (sigma2 * math.sqrt(2 * math.pi)))
        print("proposal proba", -float(pow(proposed_model_.death_param[0] - proposed_model_.death_param[1], 2)))
        print("proposal ratio",
              -float(pow(proposed_model_.death_param[0] - proposed_model_.death_param[1], 2) / (2 * pow(sigma2, 2))))
        print("proposed phi", proposed_model_.phi)
        print("current phi", current_model_.phi)
        print("ratio phi ", ((proposed_model_.phi - current_model_.phi) / 2))
        print("2nd term", (-float(
            pow(proposed_model_.death_param[0] - proposed_model_.death_param[1], 2) / (2 * pow(sigma2, 2))) - (
                                   (proposed_model_.phi - current_model_.phi) / 2)))
        print("result", (max(y_dobs) - min(y_dobs)) / (sigma2 * math.sqrt(2 * math.pi)) * math.exp(
            -float(pow(vj - vi, 2) / (2 * pow(sigma2, 2))) - (
                    (proposed_model_.phi - current_model_.phi) / 2)))
        return delta_v / (sigma2 * math.sqrt(2 * math.pi)) * math.exp(
            -(pow(vj - vi, 2) / (2 * pow(sigma2, 2))) - (
                    (proposed_model_.phi - current_model_.phi) / 2))
    else:
        # proposed_likelihood = proposed_model_.compute_likelihood()
        # current_likelihood = current_model_.compute_likelihood()
        # return prior * ((-proposed_model_.phi + current_model_.phi) / 2)
        return 0  # prior * (proposed_likelihood / current_likelihood)  # acceptance term


def compute_log_acceptance(current_model_, proposed_model_):
    # Compute prior of the proposed model (i.e. check if npa, x and y within bounds)
    prior = 1
    if proposed_model_.npa < initial_model.npa_min or proposed_model_.npa > initial_model.npa_max:
        prior = 0
    else:
        for i in range(len(proposed_model_.x)):
            if proposed_model_.x[i] > x_max or proposed_model_.x[i] < x_min or \
                    proposed_model_.y[i] > max(y_dobs) or proposed_model_.y[i] < min(y_dobs):
                prior = 0
    # print("prior", prior)
    print("perturbation", proposed_model_.curr_perturbation)
    sigma2 = 5  # standard deviation from Gaussian probability density
    delta_v = max(y_dobs) - min(y_dobs)  # uniform y distribution
    if proposed_model_.curr_perturbation == "move":
        # Compute likelihood of the proposed model
        proposed_likelihood = proposed_model_.compute_likelihood()
        current_likelihood = current_model_.compute_likelihood()
        print("proposed likelihood", proposed_likelihood)
        print("current likelihood", current_likelihood)
        print("ratio", proposed_likelihood / current_likelihood)
        # print('proposed likelihood', proposed_likelihood)
        return (current_model_.phi - proposed_model_.phi) / 2
    elif proposed_model_.curr_perturbation == "birth":
        vnp1 = proposed_model_.birth_param[0]
        vi = proposed_model_.birth_param[1]
        print("1st term", sigma2 * math.sqrt(2 * math.pi) / (max(y_dobs) - min(y_dobs)))
        print("proposal proba", (pow(vnp1 - vi, 2)))
        print("proposal ratio", (pow(vnp1 - vi, 2) / (2 * pow(sigma2, 2))))
        print("little test", (pow(vnp1 - vi, 2) / (2 * 4)))
        print("proposed phi", proposed_model_.phi)
        print("current phi", current_model_.phi)
        print("ratio phi ", ((proposed_model_.phi - current_model_.phi) / 2))
        print("2nd term", (pow(vnp1 - vi, 2) / (2 * pow(sigma2, 2))) - ((proposed_model_.phi - current_model_.phi) / 2))
        print("exp",
              math.exp((pow(vnp1 - vi, 2) / (2 * pow(sigma2, 2))) - ((proposed_model_.phi - current_model_.phi) / 2)))
        print("result", (sigma2 * math.sqrt(2 * math.pi) / (max(y_dobs) - min(y_dobs))) * math.exp(
            (pow(vnp1 - vi, 2) / (2 * pow(sigma2, 2))) - ((proposed_model_.phi - current_model_.phi) / 2)))
        return math.log(sigma2 * math.sqrt(2 * math.pi) / delta_v) + (
                (pow(vnp1 - vi, 2) / (2 * pow(sigma2, 2))) - ((proposed_model_.phi - current_model_.phi) / 2))
    elif proposed_model_.curr_perturbation == "death":
        vj = proposed_model_.death_param[0]
        vi = proposed_model_.death_param[1]
        print('vj', proposed_model_.death_param[0])
        print('vi', proposed_model_.death_param[1])
        print('death param', proposed_model_.death_param)
        print("1st term", (max(y_dobs) - min(y_dobs)) / (sigma2 * math.sqrt(2 * math.pi)))
        print("proposal proba", -float(pow(proposed_model_.death_param[0] - proposed_model_.death_param[1], 2)))
        print("proposal ratio",
              -float(pow(proposed_model_.death_param[0] - proposed_model_.death_param[1], 2) / (2 * pow(sigma2, 2))))
        print("proposed phi", proposed_model_.phi)
        print("current phi", current_model_.phi)
        print("ratio phi ", ((proposed_model_.phi - current_model_.phi) / 2))
        print("2nd term", (-float(
            pow(proposed_model_.death_param[0] - proposed_model_.death_param[1], 2) / (2 * pow(sigma2, 2))) - (
                                   (proposed_model_.phi - current_model_.phi) / 2)))
        print("result", (max(y_dobs) - min(y_dobs)) / (sigma2 * math.sqrt(2 * math.pi)) * math.exp(
            -float(pow(vj - vi, 2) / (2 * pow(sigma2, 2))) - (
                    (proposed_model_.phi - current_model_.phi) / 2)))
        return math.log(delta_v / (sigma2 * math.sqrt(2 * math.pi))) + (
                -(pow(vj - vi, 2) / (2 * pow(sigma2, 2))) - (
                (proposed_model_.phi - current_model_.phi) / 2))
    else:
        # proposed_likelihood = proposed_model_.compute_likelihood()
        # current_likelihood = current_model_.compute_likelihood()
        # return prior * ((-proposed_model_.phi + current_model_.phi) / 2)
        return 0  # prior * (proposed_likelihood / current_likelihood)  # acceptance term


def draw_fit_curve(mean_x, mean_y):
    xFit = np.arange(min(mean_x), max(mean_x), 0.01)
    if len(mean_x) >= 10:
        popt, _ = curve_fit(fifth_polynomial_regression, mean_x, mean_y)
        a, b, c, d, e, f = popt
        plt.plot(xFit, fifth_polynomial_regression(xFit, *popt), 'purple')
    elif 3 < len(mean_x) < 10:
        popt, _ = curve_fit(second_polynomial_regression, mean_x, mean_y)
        a, b, c = popt
        plt.plot(xFit, second_polynomial_regression(xFit, *popt), 'purple')
    elif 1 < len(mean_x) < 4:
        popt, _ = curve_fit(linear_regression, mean_x, mean_y)
        a, b = popt
        plt.plot(xFit, linear_regression(xFit, *popt), 'purple')


# define the true objective function
def fifth_polynomial_regression(x, a, b, c, d, e, f):
    return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + f


def second_polynomial_regression(x, a, b, c):
    return a * x + b * x ** 2 + c


def linear_regression(x, a, b):
    return a * x + b


######### Main #########
np.random.seed(2974)
# Plot true model
plt.hlines(y=0, xmin=0, xmax=1, linewidth=2, color='r', label='true model (np = 9)')
plt.vlines(x=1, ymin=0, ymax=20, linewidth=2, color='r')
plt.hlines(y=20, xmin=1, xmax=2.3, linewidth=2, color='r')
plt.vlines(x=2.3, ymin=0, ymax=20, linewidth=2, color='r')
plt.hlines(y=0, xmin=2.3, xmax=2.5, linewidth=2, color='r')
plt.vlines(x=2.5, ymin=-5, ymax=0, linewidth=2, color='r')
plt.hlines(y=-5, xmin=2.5, xmax=4, linewidth=2, color='r')
plt.vlines(x=4, ymin=-10, ymax=-5, linewidth=2, color='r')
plt.hlines(y=-10, xmin=4, xmax=6, linewidth=2, color='r')
plt.vlines(x=6, ymin=-20, ymax=-10, linewidth=2, color='r')
plt.hlines(y=-20, xmin=6, xmax=6.5, linewidth=2, color='r')
plt.vlines(x=6.5, ymin=-20, ymax=30, linewidth=2, color='r')
plt.hlines(y=30, xmin=6.5, xmax=8, linewidth=2, color='r')
plt.vlines(x=8, ymin=0, ymax=30, linewidth=2, color='r')
plt.hlines(y=0, xmin=8, xmax=9, linewidth=2, color='r')
plt.vlines(x=9, ymin=0, ymax=15, linewidth=2, color='r')
plt.hlines(y=15, xmin=9, xmax=10, linewidth=2, color='r')

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

######### RJMCMC implementation #########
# Build initial model
initial_model = Model()
initial_model.build_initial_model(x_min, x_max, y_dobs)
print(initial_model.x, initial_model.y, initial_model.npa)
initial_model.draw_lines()

# Compute phi of the initial model
initial_model.compute_phi(x_dobs, y_dobs)
# print('first likelihood', current_likelihood)

# Set RJMCMC variables
current_model = initial_model
burn_in = 10000  # length of the burn-in period
nsamples = 50000  # total number of samples
accepted_models = 0  # number of accepted models
rejected_models = 0  # number of rejected models
log_accepted_models = 0  # number of accepted models
log_rejected_models = 0  # number of rejected models
model_space = []  # model space we want to sample

for sample in range(nsamples):
    # Build proposed model with a perturbation from current model
    proposed_model = Model()
    proposed_model.build_proposed_model(current_model)

    # Compute prior of the proposed model (i.e. check if npa, x and y within bounds)
    prior = proposed_model.compute_prior()
    if prior == 0:  # if out of bounds reject the proposition
        rejected_models += 1
        log_rejected_models += 1
        print("model rejected")
        print("AND model log rejected")
        if sample >= burn_in:
            model_space.append(current_model)

    else:
        # Compute phi of the proposed model
        proposed_model.compute_phi(x_dobs, y_dobs)
        print("current model phi", current_model.phi)
        print("proposed model phi", proposed_model.phi)

        # Compute acceptance term and accept or reject it
        u = np.random.random_sample()
        log_u = math.log(u)
        print('u', u)
        print("log u", log_u)
        alpha = compute_acceptance(current_model, proposed_model)
        log_alpha = compute_log_acceptance(current_model, proposed_model)
        print('alpha', alpha)
        print("log alpha", log_alpha)
        if alpha > 1:
            alpha = 1
        if alpha >= u:  # if accepted
            # current_model = proposed_model
            accepted_models += 1
            print("model accepted")
        else:  # if rejected
            rejected_models += 1
            print("model rejected")
        if log_alpha >= log_u:  # if accepted
            current_model = proposed_model
            log_accepted_models += 1
            print("model log accepted")
        else:  # if rejected
            log_rejected_models += 1
            print("model log rejected")

        # Collect models in the chain if burn-in period has passed
        if sample >= burn_in:
            model_space.append(current_model)

print("accepted models", accepted_models)
print("rejected models", rejected_models)
print("log accepted models", log_accepted_models)
print("log rejected models", log_rejected_models)
acceptance_rate = 100 * accepted_models / nsamples
print("acceptance rate", acceptance_rate)

# Take the mean model from model space
"""
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
"""

# Compute model number of partitions and take max likelihood model
npa_number = []
model_likelihood = []
model_posterior = []
gaussian_formula = 1.0 / pow(2.0 * math.pi * sigma, len(x_dobs) / 2)  # likelihood model equation first term
for model in model_space:
    npa_number.append(model.npa)
    # model.store_lines()
    # print("model lines", model.lines)
    # model.compute_phi(x_dobs, y_dobs)
    model_likelihood.append(gaussian_formula * model.compute_likelihood())
    # print("1", gaussian_formula * model.compute_likelihood())
    # print("2", prior * (gaussian_formula * model.compute_likelihood()))
max_likelihood_model = model_likelihood.index(max(model_likelihood))
# current_model.draw_lines()
# model_space[max_likelihood_model].draw_lines()
# print("max", max_likelihood_model)
# print("max", max_posterior_model)

# print("npa list", npa_number)
print('mean', statistics.mean(npa_number))
print('std', statistics.stdev(npa_number))

# Take the mean model from model space


"""
initial_model.store_lines()
for line in range(len(initial_model.lines)):
    L1 = LineString([[initial_model.lines[line][0][0], initial_model.lines[line][0][1]],
                     [initial_model.lines[line][1][0], initial_model.lines[line][1][1]]])
    print("L1", L1)
    int_pt = L1.intersection(L2)
    print('type', type(int_pt))
    print("point", int_pt)
    print("len", len(int_pt.xy))
    if int_pt.is_empty:
        continue
    else:
        point_of_intersection_x = int_pt.x
        point_of_intersection_y = int_pt.y
        print('point x', point_of_intersection_x)
        print('point x', type(point_of_intersection_x))
        print('point x', type(float(point_of_intersection_x)))
        print('point y', point_of_intersection_y)
        x.append(point_of_intersection_x)
        y.append(point_of_intersection_y)
"""
"""
mean_model = Model()
for step in (x * 0.5 for x in range(0, 21)):
    print("step", step)
    x = []
    y = []
    L2 = LineString([[step, y_max], [step, y_min]])
    for model in model_space:  # each model
        for line in range(len(model.lines)):  # (each model line)
            L1 = LineString([[model.lines[line][0][0], model.lines[line][0][1]],
                             [model.lines[line][1][0], model.lines[line][1][1]]])
            print("L1", L1)
            int_pt = L1.intersection(L2)
            print('type', type(int_pt))
            print("point", int_pt)
            print("len", len(int_pt.xy))
            if int_pt.is_empty:
                continue
            else:
                point_of_intersection_x = int_pt.x
                point_of_intersection_y = int_pt.y
                print('point x', point_of_intersection_x)
                print('point x', type(point_of_intersection_x))
                print('point x', type(float(point_of_intersection_x)))
                print('point y', point_of_intersection_y)
                x.append(point_of_intersection_x)
                y.append(point_of_intersection_y)
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    mean_model.x.append(mean_x)
    mean_model.y.append(mean_y)
    print("mean x", mean_x)
    print("mean y", mean_y)
"""
############# Figure plot #############
# Model curves and points
plt.scatter(x_dobs, y_dobs, c='orange', label='observed data')
plt.scatter(initial_model.x, initial_model.y, c='blue', marker='s', label='initial model')
plt.scatter(current_model.x, current_model.y, c='green', marker='s', label='last accepted model')
plt.scatter(model_space[max_likelihood_model].x, model_space[max_likelihood_model].y, c='cyan', marker='s',
            label='best fit model')
"""for averaged_point in range(len(mean_model.x) - 1):
    plt.scatter(mean_model.x[averaged_point], mean_model.y[averaged_point], c='purple', marker='s')
plt.scatter(mean_model.x[len(mean_model.x) - 1], mean_model.y[len(mean_model.x) - 1], c='purple', marker='s',
            label='mean model')
# draw_fit_curve(mean_model.x, mean_model.y)
plt.scatter(mean_x, mean_y, c='purple', marker='s')"""
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend(loc='lower right')
plt.title('Regression problem')
plt.grid()

# Histograms of dimension prior and posterior probability
plot2 = plt.figure(2)
plt.hist(npa_number, range=(initial_model.npa_min, initial_model.npa_max),
         bins=(initial_model.npa_max - initial_model.npa_min), density=True, color='purple', edgecolor='black')
plt.hist(initial_model.npa_max, range=(0, initial_model.npa_max), bins=1, alpha=0.5, density=True, color='cyan')
v1 = plt.vlines(x=9, ymin=0, ymax=0.3, linewidth=2, color='r')
plt.xlabel('no. partitions')
plt.ylabel('frequency in posterior ensemble')
plt.title('p(np|d)')
plt.show()
