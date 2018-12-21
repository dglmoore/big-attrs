import main
import neet.boolean as nb
import neet.synchronous as ns
import networkx as nx
import random
import time
import unittest
import os

from datadex import DataDex
from itertools import permutations
from neet.boolean.examples import *


class Attractors(unittest.TestCase):
    def test_canary(self):
        self.assertEqual(1 + 2, 3)

    def test_builtins(self):
        nets = [s_pombe, s_cerevisiae, c_elegans, p53_dmg, p53_no_dmg, mouse_cortical_7B,
                mouse_cortical_7C, myeloid]

        k, N = 0, 0
        for net in nets:
            name = net.metadata['name']
            g = net.to_networkx_graph()
            print()
            print("{}: {}".format(name, net.size))
            print("  Modules: {}".format(list(map(len, nx.strongly_connected_components(g)))))
            print("  DAG: {}".format(list(nx.condensation(g).edges())))

            start = time.time()
            got = main.attractors(net)
            stop = time.time()
            modular = stop - start
            print("  Modular Time:     {}s".format(modular))

            start = time.time()
            expected = ns.attractors(net)
            stop = time.time()
            brute_force = stop - start
            print("  Brute Force Time: {}s".format(brute_force))
            print("  Factor: {}".format(modular / brute_force))

            k += modular / brute_force
            N += 1

            self.assertUnorderedEqual(expected, got, msg=name)
            self.assertUnorderedEqual(got, expected, msg=name)

        self.assertTrue(k / N < 1.0, msg='algorithm is slower than brute force on average')


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

    def test_moderate_grns(self):
        k, N = 0, 0

        dex = DataDex()
        data = dex.select(['name', 'filename'], ['is_biological', 'node_count <= 22'])
        for name, filename in data:
            exp_filename = os.path.join(filename, 'expressions.txt')
            ext_filename = os.path.join(filename, 'external.txt')
            net = nb.LogicNetwork.read_logic(exp_filename, ext_filename)
            g = net.to_networkx_graph()
            print()
            print("{}: {}".format(name, net.size))
            print("  Modules: {}".format(list(map(len, nx.strongly_connected_components(g)))))
            print("  DAG: {}".format(list(nx.condensation(g).edges())))

            start = time.time()
            got = main.attractors(net)
            stop = time.time()
            modular = stop - start
            print("  Modular Time:     {}s".format(modular))

            start = time.time()
            expected = ns.attractors(net)
            stop = time.time()
            brute_force = stop - start
            print("  Brute Force Time: {}s".format(brute_force))
            print("  Factor: {}".format(modular / brute_force))

            k += modular / brute_force
            N += 1

            self.assertUnorderedEqual(expected, got, msg=name)
            self.assertUnorderedEqual(got, expected, msg=name)

        print(k / N)
        self.assertTrue(k / N < 1.0, msg='algorithm is slower than brute force on average')

    #  def test_iron_aquisition(self):
    #      name = 'Iron Acquisition And Oxidative Stress Response In Aspergillus Fumigatus.'
    #      filename = DataDex().select('filename', 'name is "{}"'.format(name))[0]
    #      exp_filename = os.path.join(filename, 'expressions.txt')
    #      ext_filename = os.path.join(filename, 'external.txt')
    #      net = nb.LogicNetwork.read_logic(exp_filename, ext_filename)
    #      g = net.to_networkx_graph()
    #      print()
    #      print("{}: {}".format(name, net.size))
    #      print("  Modules: {}".format(list(map(len, nx.strongly_connected_components(g)))))
    #      print("  DAG: {}".format(list(nx.condensation(g).edges())))
    #
    #      start = time.time()
    #      got = main.attractors(net)
    #      stop = time.time()
    #      modular = stop - start
    #      print("  Modular Time:     {}s".format(modular))
    #
    #      start = time.time()
    #      expected = ns.attractors(net)
    #      stop = time.time()
    #      brute_force = stop - start
    #      print("  Brute Force Time: {}s".format(brute_force))
    #      print("  Factor: {}".format(modular / brute_force))
    #
    #      self.assertUnorderedEqual(expected, got, msg=name)
    #      self.assertUnorderedEqual(got, expected, msg=name)

    def assertUnorderedEqual(self, first, second, msg=None):
        for a in first:
            found = False
            for b in second:
                if tuple(a) in permutations(b):
                    found = True
                    break
            self.assertTrue(found, msg="did not find {} in {}".format(a, second))

