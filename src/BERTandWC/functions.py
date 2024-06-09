import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
import os
import fnmatch
from os.path import join, dirname, abspath
from yaml.constructor import SafeConstructor
import sys
tf.keras.backend.set_floatx('float32')



def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

    # create constructor for !path tag
    def construct_path(self, node):
        return os.path.normpath(self.construct_scalar(node))


PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)

PrettySafeLoader.add_constructor(
    '!path',
    PrettySafeLoader.construct_path)


def LoadConfig(config_name):
    """
    This function loads the configuration file from the config folder. If hyper is not None, it will replace the
    hyperparameters in the configuration file with the ones in hyper
    :param config_name: The name of the configuration file
    :return: The configuration dictionary
    """
    configuration_file = find(config_name, join(dirname(abspath(__file__))))[0]
    with open(configuration_file, 'r') as c:
        configuration = yaml.load(c, Loader=PrettySafeLoader)

    return configuration


class DatasetMaker:
    def __init__(self, config: dict, space: str = 'direct'):
        self.config = config
        self.net_size = config['Network']['size']
        self.number_classes = config['Dataset']['classes']
        self.space = space
        self.attractor_matrix = self.attractors_maker()

    def attractor_shape_maker(self):
        # Create the stable attractors structure
        attractor_list_x = []
        attractor_list_y = []
        for i in range(self.number_classes):
            attractor = np.ones(self.net_size)
            # Set the number of ones in the attractor in non overlapping regions
            attractor[i * (self.net_size // (self.number_classes + 2)): (i + 1) * (
                        self.net_size // (self.number_classes + 2))] = 0
            attractor_list_x.append(attractor.copy())  # Add the attractor to the list for x component
        attractor_list_y.append(np.ones(self.net_size))  # Add the attractor to the list for y component
        return attractor_list_x, attractor_list_y

    def attractors_maker(self, concatenated: bool = False, plot: bool = True):
        """
        Create a list of attractors for the dataset. They are self.number_classes + 1, because the last one is the unstable
        attractor. The first self.number_classes are the stable attractors and, at first, are non overlapping binary vectors of size
        self.net_size (the ones are always in different regions). The number of ones is self.net_size // self.number_classes.
        The last one is the unstable attractor and is a vector of self.net_size // 2 ones placed at random.
        The values of 0 and 1 are then sobstituted with the corresponding values of the fixed points given in the config
        file (config['FixedPoints']['Stable']): the latter returns a dictionary containing the tuples x,y. The x is assigned to 0 and y to 1.
        Same for the unstable attractor, but the values are in config['FixedPoints']['Unstable']

        :return: The matrix containing as columns the attractors
        """

        # Create the stable attractors structure
        attractor_list_x, attractor_list_y = self.attractor_shape_maker()
        data=[]
        if self.config.get('FixedPoints') is None:
            for i, tmp in enumerate(attractor_list_x):
                tmp[tmp == 0] = 0.213590341241989722931293727015145
                tmp[tmp == 1] = 0.37326898463326614599362307391132
                data.append(np.asarray(tmp))
            return np.array(data).T

        else:
            # Substitute the values of 0 and 1 with the corresponding values of the fixed points
            stable_points = np.array([i for i in self.config['FixedPoints'][
                'Stable'].values()])  # Matrix of shape (# Fixed Points, 2), each row is a fixed point (x,y)

            for i, tmp in enumerate(attractor_list_x):
                tmp[tmp == 0] = stable_points[0, 0]
                tmp[tmp == 1] = stable_points[1, 0]

            for i, tmp in enumerate(attractor_list_y):
                tmp[tmp == 0] = stable_points[0, 1]
                tmp[tmp == 1] = stable_points[1, 1]

            if plot:
                fig, axs = plt.subplots(2, self.number_classes)
                if self.number_classes == 1:
                    axs = axs[:, np.newaxis]

                for i, (attractor_x, attractor_y) in enumerate(zip(attractor_list_x, attractor_list_y)):
                    ax = axs[0, i]
                    plt.colorbar(ax.imshow(attractor_x.reshape(int(np.sqrt(self.net_size)), -1), cmap='viridis'), ax=ax,
                                 fraction=0.046)
                    if i == 0:
                        ax.set_ylabel('Attractors X')
                    else:
                        ax.axis('off')
                    ax = axs[1, i]
                    plt.colorbar(ax.imshow(attractor_y.reshape(int(np.sqrt(self.net_size)), -1), cmap='viridis'), ax=ax,
                                 fraction=0.046)
                    if i == 0:
                        ax.set_ylabel('Attractors Y')
                    else:
                        ax.axis('off')
                plt.tight_layout()
                plt.savefig('attraactors.jpg')
                plt.show()

        # Create the homogeneous unstable attractor structure
        unstable_points = np.array([i for i in self.config['FixedPoints']['Unstable'].values()])

        unstable_attractor_x = np.ones(self.net_size) * unstable_points[0, 0]
        unstable_attractor_y = np.ones(self.net_size) * unstable_points[0, 1]

        attractor_list = attractor_list_x + attractor_list_y + [unstable_attractor_x] + [unstable_attractor_y]

        attractor_matrix = np.array(attractor_list).T

        if concatenated:
            # Concatenate the x and y components of the attractors along the rows
            attractor_matrix_stable = np.concatenate((np.array(attractor_list_x).T, np.array(attractor_list_y).T),
                                                     axis=0)
            attractor_matrix_unstable = np.concatenate((unstable_attractor_x, unstable_attractor_y), axis=0)
            return np.concatenate((attractor_matrix_stable, attractor_matrix_unstable[:, np.newaxis]), axis=1)

        return attractor_matrix
 