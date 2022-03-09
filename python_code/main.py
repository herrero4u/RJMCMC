"""
This script recreates the regression example presented in "Transdimensional inference in the geosciences" paper
(Sambridge et al. 2013) using transdimensional inversions with a Voronoi nuclei parameterization.
Through a random initial prior model, observed data, and RJMCMC iterations, perturbations as moves, births and deaths of
Voronoi nuclei are proposed to explore the model space and converge to likely solutions in different dimensions.
At the end, it is possible to extract a smooth mean model that aims to fit the true model as fine as possible.
The y coordinate of model partitions is the parameter of interest in the Bayesian inversion. The model dimension k
concerns the number of model partitions, each being defined by Voronoi nuclei.
"""

__author__ = "Julien Herrero"
__contact__ = "julien.herrero@univ-lorraine.fr"
__copyright__ = "RING Team"
__date__ = "2022-03-01"
__version__ = "1"

import sys
import math
import numpy as np
import matplotlib.pyplot as plt


class Model:
    """Store model unknowns (Voronoi nuclei number and coordinates) and parameterization
    Attributes:
        x (float list): x coordinate of each nucleus
        y (float list): y coordinate of each nucleus
        npa (int): number of partitions
        npa_min (int): minimal possible number of partitions
        npa_max (int): maximal possible number of partitions
        phi (float): misfit function which quantifies the agreement between simulated and observed data
        birth_param (tuple): store nucleus birth parameterization
        death_param (tuple): store nucleus death parameterization
        curr_perturbation (str): type of perturbation at the current iteration
    """

    def __init__(self):
        """Model class constructor
        """
        self.x = []
        self.y = []
        self.npa_min = int
        self.npa_max = int
        self.npa = int
        self.phi = float
        self.birth_param = tuple
        self.death_param = tuple
        self.curr_perturbation = str

    def build_initial_model(self, boundaries, y_dobs):
        """Build the initial model of the chain from a uniform prior distribution
        :param boundaries: x-y minimal and maximal coordinates in the field
        :param y_dobs: y coordinates of observed data used as y min-max coordinates
        :type boundaries: int numpy array(2,2)
        :type y_dobs: float list
        """
        self.npa_min = 1  # min number of partitions
        self.npa_max = 50  # max number of partitions
        self.npa = np.random.randint(self.npa_min, self.npa_max + 1)  # number of partitions
        x_min = boundaries[0][0]
        x_max = boundaries[1][0]
        for k in range(self.npa):
            self.x.append(np.random.uniform(x_min, x_max))  # x prior distribution
            self.y.append(np.random.uniform(min(y_dobs), max(y_dobs)))  # y prior distribution

    def build_proposed_model(self, current_model, boundaries):
        """Build a proposed model from the current model with a random perturbation (birth, death, and move)
        :param current_model: model used in the chain at the current iteration
        :param boundaries: x-y minimal and maximal coordinates in the field
        :type current_model: Model
        :type boundaries: int numpy array(2,2)
        """
        self.npa = current_model.npa
        self.npa_min = current_model.npa_min
        self.npa_max = current_model.npa_max
        perturb_type = np.random.random_sample()  # random number to choose the perturbation type to apply
        if perturb_type < 0.33:
            self.curr_perturbation = "move"
            self.move(current_model)
        elif 0.33 <= perturb_type <= 0.66:
            self.curr_perturbation = "birth"
            self.birth(current_model, boundaries)
        else:
            self.curr_perturbation = "death"
            self.death(current_model)

    def extract_mean_model(self, model_space):
        """Extract the mean model according to a chosen step along x-axis by calculating the y coordinate mean of all
        models stored in the model space for the given x coordinate. The y values are found with the closest Voronoi
        nucleus from the step
        Notes: it is possible to also extract the map error, by computing the y standard deviation
        :param model_space: space sampled of post burn-in models to approximate the posterior probability density
        :type model_space: numpy array(n_samples - burn-in) of Model objects
        """
        for x_step in (x * 0.5 for x in range(0, 21)):
            stacked_y = []  # partition y values of all models at a given spatial step of the grid
            for model in model_space:
                distance = np.empty(model.npa, dtype=float)
                for nucleus in range(model.npa - 1):
                    distance[nucleus] = abs(x_step - model.x[nucleus])
                index_min_dist = (np.where(distance == min(distance)))[0][0]
                stacked_y.append(model.y[index_min_dist])
            self.x.append(x_step)
            self.y.append(np.mean(stacked_y))

    def move(self, current_model):
        """Propose a random move of nuclei as a perturbation of the model
        :param current_model: model used in the chain at the current iteration
        :type current_model: Model
        """
        move_type = np.random.random_sample()  # random number to choice the move perturbation to apply
        if move_type < 0.33:  # move all nuclei (gaussian perturbation)
            for k in range(self.npa):
                self.x.append(np.random.normal(current_model.x[k], 0.3))
                self.y.append(np.random.normal(current_model.y[k], 5))
        elif 0.33 <= move_type <= 0.66:  # move one nucleus in x & y axes (gaussian perturbation)
            for k in range(self.npa):
                self.x.append(current_model.x[k])
                self.y.append(current_model.y[k])
            move_index = self.x.index(np.random.choice(self.x))  # index of the random point to move
            self.x[move_index] = np.random.normal(self.x[move_index], 0.5)
            self.y[move_index] = np.random.normal(self.y[move_index], 10)
        else:  # move one nucleus in x-axis (gaussian perturbation)
            for k in range(self.npa):
                self.x.append(current_model.x[k])
                self.y.append(current_model.y[k])
            move_index = self.x.index(np.random.choice(self.x))  # index of the random point to move
            self.x[move_index] = np.random.normal(self.x[move_index], 1)

    def full_rand_birth(self, current_model, boundaries):
        """Propose a full random nucleus birth from uniform prior distribution
        :param current_model: model used in the chain at the current iteration
        :param boundaries: x-y minimal and maximal coordinates in the field
        :type current_model: Model
        :type boundaries: int numpy array(2,2)
        """
        self.npa += 1  # increasing number of partitions
        for k in range(self.npa - 1):
            self.x.append(current_model.x[k])
            self.y.append(current_model.y[k])
        x_min = boundaries[0][0]
        x_max = boundaries[1][0]
        y_min = boundaries[0][1]
        y_max = boundaries[1][1]
        self.x.append(np.random.uniform(x_min, x_max))  # birth from x prior distribution
        self.y.append(np.random.uniform(y_min, y_max))  # birth from y prior distribution

    def full_rand_death(self, current_model):
        """Propose a full random nucleus death from uniform prior distribution
        :param current_model: model used in the chain at the current iteration
        :type current_model: Model
        """
        if self.npa > 1:  # the model dimension can not be less than 1
            self.npa -= 1  # decreasing number of partitions
            for k in range(self.npa + 1):
                self.x.append(current_model.x[k])
                self.y.append(current_model.y[k])
            death_index = self.x.index(np.random.choice(self.x))  # index of the random point to delete
            del self.x[death_index]  # death from x prior distribution
            del self.y[death_index]  # death from y prior distribution
        else:
            self.curr_perturbation = "nothing"
            print('npa = ', self.npa, 'so nothing is done')
            for k in range(self.npa):
                self.x.append(current_model.x[k])
                self.y.append(current_model.y[k])

    def birth(self, current_model, boundaries):
        """Propose a nucleus birth distributed from a gaussian density function
        :param current_model: model used in the chain at the current iteration
        :param boundaries: x-y minimal and maximal coordinates in the field
        :type current_model: Model
        :type boundaries: int numpy array(2,2)
        """
        self.npa += 1  # increasing number of partitions
        sigma2 = 10.  # standard deviation from Gaussian proposal probability density
        for k in range(self.npa - 1):
            self.x.append(current_model.x[k])
            self.y.append(current_model.y[k])
        x_min = boundaries[0][0]
        x_max = boundaries[1][0]
        self.x.append(np.random.uniform(x_min, x_max))  # x random birth
        distance = []
        for nucleus in range(self.npa - 1):
            distance.append(abs(self.x[self.npa - 1] - self.x[nucleus]))
        index_min_dist = distance.index(min(distance))
        vi = self.y[index_min_dist]  # current y value (from birth closest nucleus)
        self.y.append(np.random.normal(vi, sigma2))  # y birth from gaussian proposal probability
        vnp1 = self.y[self.npa - 1]  # new y parameter value
        self.birth_param = (vnp1, vi)

    def death(self, current_model):
        """Propose a nucleus death distributed from a gaussian density function
        :param current_model: model used in the chain at the current iteration
        :type current_model: Model
        """
        if self.npa > 1:  # the model dimension can not be less than 1
            self.npa -= 1  # decreasing number of partitions
            for k in range(self.npa + 1):
                self.x.append(current_model.x[k])
                self.y.append(current_model.y[k])
            death_index = self.x.index(np.random.choice(self.x))  # index of the random point to delete
            point_to_kill = self.x[death_index]  # value of the nucleus to be killed
            temp_x = []  # temp is only used to compute the index of the closest point from the killed nucleus
            for nucleus in range(len(self.x)):
                temp_x.append(self.x[nucleus])
            temp_x[death_index] = 100000
            del self.x[death_index]  # x random death
            distance = []
            for nucleus in range(len(self.y)):
                distance.append(abs(point_to_kill - temp_x[nucleus]))
            index_min_dist = distance.index(min(distance))
            vi = self.y[death_index]  # current y value which is about to be killed
            vj = self.y[index_min_dist]  # y value at the point in the new partition (from death closest nucleus)
            self.death_param = (vj, vi)
            del self.y[death_index]  # y death from a reverse gaussian proposal probability
        else:
            self.curr_perturbation = "nothing"
            for k in range(self.npa):
                self.x.append(current_model.x[k])
                self.y.append(current_model.y[k])

    def compute_prior(self, boundaries, y_dobs):
        """Check if current model is in prior model bounds (i.e., npa, x and y within bounds) and if not prior
        probability is set to 0, meaning that the model will be automatically rejected in a MCMC chain
        :param boundaries: x-y minimal and maximal coordinates in the field
        :param y_dobs: y coordinates of observed data used as y min-max coordinates
        :type boundaries: int numpy array(2,2)
        :type y_dobs: float list
        :return: int prior probability after checking if the model is within boundaries (0 or 1)
        """
        prior = 1
        if self.npa < self.npa_min or self.npa > self.npa_max:
            prior = 0
        else:
            x_min = boundaries[0][0]
            x_max = boundaries[1][0]
            for nucleus in range(len(self.x)):
                if self.x[nucleus] > x_max or self.x[nucleus] < x_min or \
                        self.y[nucleus] > max(y_dobs) or self.y[nucleus] < min(y_dobs):
                    prior = 0
        return prior

    def compute_likelihood(self, sigma, x_dobs):
        """Compute likelihood of the model from misfit function
        :param sigma: estimated (or known) standard deviation of the data noise (assumed uncorrelated)
        :param x_dobs: x coordinates of observed data
        :type sigma: float
        :type x_dobs: float list
        :return: float result of the likelihood probability
        """
        gaussian_term = 1.0 / pow(2.0 * math.pi * sigma, len(x_dobs) / 2)  # likelihood model equation first term
        return gaussian_term * (math.exp(-(1 / 2) * self.phi))

    def compute_phi(self, sigma, x_dobs, y_dobs):
        """Compute the function which quantifies the agreement between estimated and observed data,
        based on a simple least squares misfit
        :param sigma: estimated (or known) standard deviation of the data noise (assumed uncorrelated)
        :param x_dobs: x coordinates of observed data
        :param y_dobs: y coordinates of observed data
        :type sigma: float
        :type x_dobs: float list
        :type y_dobs: float list
        """
        self.phi = 0
        for j in range(len(x_dobs)):
            distance = []
            for nucleus in range(len(self.x)):
                distance.append(abs(x_dobs[j] - self.x[nucleus]))
            index_min_dist = distance.index(min(distance))  # index of distance with the closest model point
            self.phi += pow(y_dobs[j] - self.y[index_min_dist], 2) / pow(sigma, 2)

    def draw_lines(self, boundaries, color):
        """Draw model partitions (lines) from coordinates of Voronoi nuclei
        :param boundaries: x-y minimal and maximal coordinates in the field
        :param color: color of lines to be plotted
        :type boundaries: int numpy array(2,2)
        :type color: str
        """
        xy_coord = []  # list of tuples gathering x and y nucleus coordinates
        for nucleus in range(len(self.x)):
            xy_coord.append((self.x[nucleus], self.y[nucleus]))
        sorted_list = sorted(xy_coord, key=lambda xy_: xy_[0])  # sort x-y coordinates by x
        for nucleus in range(len(self.x)):
            self.x[nucleus] = sorted_list[nucleus][0]
            self.y[nucleus] = sorted_list[nucleus][1]
        x_min_line = boundaries[0][0]
        change_point = 0
        for k in range(1, len(self.x)):
            change_point = (self.x[k] + self.x[k - 1]) / 2
            plt.hlines(y=self.y[k - 1], xmin=x_min_line, xmax=change_point, linewidth=2, color=color)
            plt.vlines(x=change_point, ymin=self.y[k - 1], ymax=self.y[k], color=color)
            x_min_line = change_point
        x_max = boundaries[1][0]
        plt.hlines(y=self.y[len(self.y) - 1], xmin=change_point, xmax=x_max)


def compute_true_model():
    """Plot true model partitions"""
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


def create_noisy_points(sigma, x_dobs, y_dobs):
    """Create noisy points along x coordinate from the true model partitions
    :param sigma: estimated (or known) standard deviation of the data noise (assumed uncorrelated)
    :param x_dobs: x coordinates of observed data
    :param y_dobs: y coordinates of observed data
    :type sigma: float
    :type x_dobs: float list
    :type y_dobs: float list
    """
    sum_x = 0.0
    step = 0.1  # observed data sampling on x
    prev_i = 0
    true_x = np.array([10, 23, 25, 40, 60, 65, 80, 90, 100])
    true_y = np.array([0, 20, 0, -5, -10, -20, 30, 0, 15])
    for i in range(len(true_x)):
        for j in range(prev_i, true_x[i]):
            sum_x += step
            x_dobs.append(sum_x)
            s = np.random.normal(true_y[i], sigma)  # add gaussian noise on points
            y_dobs.append(s)
        prev_i = true_x[i]


def compute_acceptance(current_model, proposed_model, y_dobs):
    """Compute acceptance probability of the perturbed model given the current model
    :param current_model: model used in the chain at the current iteration
    :param proposed_model: model proposed in the chain at the current iteration
    :param y_dobs: y coordinates of observed data
    :type current_model: Model
    :type proposed_model: Model
    :type y_dobs: float list
    :return: float acceptance term computed with expression that depends on the type of perturbation
    applied on the proposed model
    """
    sigma2 = 10.  # standard deviation from Gaussian proposal probability density
    delta_v = max(y_dobs) - min(y_dobs)  # y uniform distribution
    if proposed_model.curr_perturbation == "move":
        return (current_model.phi - proposed_model.phi) / 2
    elif proposed_model.curr_perturbation == "birth":
        vnp1 = proposed_model.birth_param[0]
        vi = proposed_model.birth_param[1]
        return math.log(sigma2 * math.sqrt(2 * math.pi) / delta_v) + (
                (pow(vnp1 - vi, 2) / (2 * pow(sigma2, 2))) - ((proposed_model.phi - current_model.phi) / 2))
    elif proposed_model.curr_perturbation == "death":
        vj = proposed_model.death_param[0]
        vi = proposed_model.death_param[1]
        return math.log(delta_v / (sigma2 * math.sqrt(2 * math.pi))) + (
                -(pow(vj - vi, 2) / (2 * pow(sigma2, 2))) - ((proposed_model.phi - current_model.phi) / 2))
    else:
        return 0


def extract_best_fit_model(model_space, sigma, x_dobs):
    """Find and extract best fit model over the model space sampling
    :param model_space: space sampled of post burn-in models to approximate the posterior probability density
    :param sigma: estimated (or known) standard deviation of the data noise (assumed uncorrelated)
    :param x_dobs: x coordinates of observed data
    :type model_space: numpy array(n_samples - burn-in) of Model objects
    :type sigma: float
    :type x_dobs: float list
    :return: max likelihood model from model_space array
    """
    model_likelihood = np.empty(len(model_space), dtype=float)
    for model_index, model in enumerate(model_space):
        model_likelihood[model_index] = model.compute_likelihood(sigma, x_dobs)
    max_likelihood_model_index = (np.where(model_likelihood == max(model_likelihood)))[0][0]
    max_likelihood_model = model_space[max_likelihood_model_index]
    return max_likelihood_model


def compute_npa_numbers(model_space):
    """Store each model number of partitions to be able to visualize the posterior probability density
     of model dimensions P(np|d)
    :param model_space: space sampled of post burn-in models to approximate the posterior probability density
    :type model_space: numpy array(n_samples - burn-in) of Model objects
    :return: int array storing each model dimension (number of partitions)
    """
    npa_number = np.empty(len(model_space), dtype=int)
    for model_index, model in enumerate(model_space):
        npa_number[model_index] = model.npa
    print("\nmodel space npa mean", np.mean(npa_number))
    print("model space npa standard deviation", np.std(npa_number))
    return npa_number


def main():
    np.random.seed(4)

    # Create observed data from the solution
    compute_true_model()
    x_dobs = []  # x coordinates of observed data
    y_dobs = []  # y coordinates of observed data
    sigma = 10.  # standard deviation
    create_noisy_points(sigma, x_dobs, y_dobs)

    # Field boundaries
    x_min = 0
    x_max = 10
    y_min = -60
    y_max = 60
    boundaries = np.array(([x_min, y_min], [x_max, y_max]))

    # Build initial model
    initial_model = Model()
    initial_model.build_initial_model(boundaries, y_dobs)
    print("initial model:\nx =", initial_model.x, "\ny =", initial_model.y, "\nnumber of partitions =",
          initial_model.npa)
    initial_model.draw_lines(boundaries, 'b')
    initial_model.compute_phi(sigma, x_dobs, y_dobs)  # initial model misfit

    # Set RJMCMC variables
    current_model = initial_model
    burn_in = 10000  # length of the burn-in period
    n_samples = 50000  # total number of samples
    accepted_models = 0  # number of accepted models
    rejected_models = 0  # number of rejected models
    model_space = np.empty(n_samples - burn_in, dtype=Model)

    for sample in range(n_samples):  # RJMCMC iterations
        # Build proposed model with a perturbation from current model
        proposed_model = Model()
        proposed_model.build_proposed_model(current_model, boundaries)

        # Compute prior of the proposed model (i.e. check if npa, x and y within bounds)
        prior = proposed_model.compute_prior(boundaries, y_dobs)
        if prior == 0:  # if out of bounds reject the proposition
            rejected_models += 1
            if sample >= burn_in:  # store current model in the chain if burn-in period has passed
                model_space[sample - burn_in] = current_model

        else:
            # Compute acceptance term and accept or reject it
            proposed_model.compute_phi(sigma, x_dobs, y_dobs)  # proposed model misfit
            u = np.random.random_sample()
            log_u = math.log(u)
            alpha = compute_acceptance(current_model, proposed_model, y_dobs)
            if alpha >= log_u:  # if accepted
                current_model = proposed_model
                accepted_models += 1
            else:  # if rejected
                rejected_models += 1

            # Collect models in the chain if burn-in period has passed
            if sample >= burn_in:
                model_space[sample - burn_in] = current_model

    print("\naccepted models", accepted_models)
    print("rejected models", rejected_models)
    acceptance_rate = 100 * accepted_models / n_samples
    print("acceptance rate", acceptance_rate)

    # Compute model number of partitions and extract best fit model
    npa_number = compute_npa_numbers(model_space)
    max_likelihood_model = extract_best_fit_model(model_space, sigma, x_dobs)
    max_likelihood_model.draw_lines(boundaries, 'green')

    # Take the mean model from model space
    mean_model = Model()
    mean_model.extract_mean_model(model_space)

    # Figure plot

    # Model curves and points
    plt.scatter(x_dobs, y_dobs, c='orange', label='observed data')
    plt.scatter(initial_model.x, initial_model.y, c='blue', marker='s', label='initial model')
    plt.scatter(max_likelihood_model.x, max_likelihood_model.y, c='green', marker='s', label='best fit model')
    plt.scatter(mean_model.x, mean_model.y, c='purple', marker='o', label='mean model')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('x')
    plt.ylabel('y', rotation=0)
    plt.legend(loc='lower right')
    plt.title('Regression problem')
    plt.grid()

    # Histograms of prior and posterior probabilities of model dimensions
    plt.figure(2)
    plt.hist(npa_number, range=(initial_model.npa_min, initial_model.npa_max), density=True, color='purple',
             edgecolor='black', bins=(initial_model.npa_max - initial_model.npa_min), label='posterior PDF')
    plt.hist(initial_model.npa_max, range=(0, initial_model.npa_max), density=True, color='cyan', bins=1, alpha=0.5,
             label='prior PDF')
    plt.vlines(x=9, ymin=0, ymax=0.3, linewidth=2, color='r', label='true model dimension')
    plt.xlabel('no. partitions')
    plt.ylabel('frequency in posterior ensemble')
    plt.legend(loc='upper right')
    plt.title('p(np|d)')
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
