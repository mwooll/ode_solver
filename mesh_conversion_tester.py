#mesh_conversion_tester
import numpy as np
from unittest import TestCase, main

from mesh_conversion import downsizing, augmentation

class TestNodes(TestCase):
    """
    Tests concerning the conversion of the nodes
    note: the boundary conditions and the values have no influence on the nodes
    the "on_{number}_{number}" at the end of the testnames signifies the interval used
    """

    """1. interval = [0,1]"""
    def test_downsizing_nodes_once_on_zero_one(self):
        interval  = [0, 1]
        nodes     = np.linspace(interval[0], interval[1], 9)
        values    = np.ones(7)

        down_nodes, down_values, down_size = downsizing(nodes, values, 4)
        actual    = list(down_nodes)
        expected  = list(np.linspace(interval[0], interval[1], 5))
        self.assertListEqual(actual, expected)

    def test_downsizing_nodes_twice_on_zero_one(self):
        interval  = [0, 1]
        nodes     = np.linspace(interval[0], interval[1], 9)
        values    = np.ones(7)

        down_nodes, down_values, down_size = downsizing(nodes, values, 2)
        actual    = list(down_nodes)
        expected  = list(np.linspace(interval[0], interval[1], 3))
        self.assertListEqual(actual, expected)

    def test_augment_nodes_once_on_zero_one(self):
        interval  = [0, 1]
        nodes     = np.linspace(interval[0], interval[1], 5)
        values    = np.ones(3)

        augm_nodes, augm_values, augm_size = augmentation(nodes, values, 16)
        actual    = list(augm_nodes)
        expected  = list(np.linspace(interval[0], interval[1], 17))
        self.assertListEqual(actual, expected)

    def test_augment_nodes_twice_on_zero_one(self):
        interval  = [0, 1]
        nodes     = np.linspace(interval[0], interval[1], 5)
        values    = np.ones(3)

        augm_nodes, augm_values, augm_size = augmentation(nodes, values, 16)
        actual    = list(augm_nodes)
        expected  = list(np.linspace(interval[0], interval[1], 17))
        self.assertListEqual(actual, expected)

    """2. variable interval"""
    def test_downsizing_nodes_once_on_minus_one_ten(self):
        interval  = [-1, 10]
        nodes     = np.linspace(interval[0], interval[1], 33)
        values    = np.ones(31)

        down_nodes, down_values, down_size = downsizing(nodes, values, 16)
        actual    = list(down_nodes)
        expected  = list(np.linspace(interval[0], interval[1], 17))
        self.assertListEqual(actual, expected)

    def test_downsizing_nodes_on_twice_zero_two(self):
        interval  = [0, 2]
        nodes     = np.linspace(interval[0], interval[1], 9)
        values    = np.ones(7)

        down_nodes, down_values, down_size = downsizing(nodes, values, 2)
        actual    = list(down_nodes)
        expected  = list(np.linspace(interval[0], interval[1], 3))
        self.assertListEqual(actual, expected)

    def test_augment_nodes_once_on_minus_one_ten(self):
        interval  = [-1, 10]
        nodes     = np.linspace(interval[0], interval[1], 9)
        values    = np.ones(7)

        augm_nodes, augm_values, augm_size = augmentation(nodes, values, 32)
        actual    = list(augm_nodes)
        expected  = list(np.linspace(interval[0], interval[1], 33))
        self.assertListEqual(actual, expected)    

    def test_augment_nodes_twice_on_zero_three(self):
        interval  = [0, 3]
        nodes     = np.linspace(interval[0], interval[1], 9)
        values    = np.ones(7)

        augm_nodes, augm_values, augm_size = augmentation(nodes, values, 32)
        actual    = list(augm_nodes)
        expected  = list(np.linspace(interval[0], interval[1], 33))
        self.assertListEqual(actual, expected)


class TestValues(TestCase):
    """
    Tests concerning the conversion of values
    note: the length of the interval has no influence on the values
    """

    """equidistant partition"""
    def test_downsizing_values_once(self):
        nodes     = np.linspace(0, 1, 9)
        values    = [0, 1, 0, 1, 0, 1, 0]

        down_nodes, down_values, down_size = downsizing(nodes, values, 4)
        actual    = list(down_values)
        expected  = [1/2, 1, 1/2]
        self.assertListEqual(actual, expected)

    def test_downsizing_values_twice(self):
        nodes     = np.linspace(0, 1, 9)
        values    = np.ones(7)

        down_nodes, down_values, down_size = downsizing(nodes, values, 2)
        actual    = [down_values]
        expected  = [3.5]
        self.assertListEqual(actual, expected)

    def test_augment_values_once(self):
        nodes     = np.linspace(0, 1, 3)
        values    = [1]

        augm_nodes, augm_values, augm_size = augmentation(nodes, values, 4)
        actual    = list(augm_values)
        expected  = [1/2, 1, 1/2]
        self.assertListEqual(actual, expected)


def test_meshes():
    main()

if __name__ == "__main__":
    test_meshes()