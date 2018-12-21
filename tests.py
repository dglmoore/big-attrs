import main
import neet.boolean as nb
import neet.synchronous as ns
import unittest
import networkx as nx
import time
import random

from itertools import permutations
from neet.boolean.examples import *


class Attractors(unittest.TestCase):
    def test_canary(self):
        self.assertEqual(1 + 2, 3)

    def test_builtins(self):
        nets = [s_pombe, s_cerevisiae, c_elegans, mouse_cortical_7B, mouse_cortical_7C, myeloid]
        for net in nets:
            expect = ns.attractors(net)
            got = main.attractors(net)
            self.assertUnorderedEqual(expect, got, net.metadata['name'])

    def test_complex(self):
        net = nb.WTNetwork([
                [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0,-1, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [ 1, 0,-1,-1, 1, 1, 0, 0, 0, 0],
                [ 0,-1, 0,-1, 1, 1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0,-1, 1, 0, 0],
                [ 0, 0, 0, 0, 1, 0,-1, 1, 0, 0],
                [ 0, 0, 0, 0, 0, 1, 0, 0,-1,-1],
                [ 0, 0, 0, 0,-1, 0, 0, 0, 1, 0],
            ])

        g = net.to_networkx_graph()
        self.assertEqual([[8, 9], [6, 7], [4, 5], [0, 1], [2, 3]],
                list(map(list, nx.strongly_connected_components(g))))
        self.assertEqual([(2, 1), (2, 0), (3, 2), (4, 2)],
                list(nx.condensation(g).edges))

        expect = ns.attractors(net)

        got = main.attractors(net)

        self.assertUnorderedEqual(expect, got)

    def assertUnorderedEqual(self, first, second, msg=None):
        self.assertEqual(len(first), len(second), msg)
        for a in first:
            found = False
            for b in second:
                if tuple(a) in permutations(b):
                    found = True
                    break
            self.assertTrue(found, msg="did not find {} in {}".format(a, second))

