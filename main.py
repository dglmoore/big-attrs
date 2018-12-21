import numpy as np
import networkx as nx

from functools import reduce
from neet.interfaces import is_network, is_fixed_sized

def subspace(subgraph, size=None, dynamic_values=None):
    subgraph = list(subgraph)
    subgraph.sort()

    if dynamic_values is None:
        state = [0] * size
        yield state[:]
        i = 0
        while i != len(subgraph):
            node = subgraph[i]
            if state[node] + 1 < 2:
                state[node] += 1
                for j in range(i):
                    state[subgraph[j]] = 0
                i = 0
                yield state[:]
            else:
                i += 1
    else:
        parent_nodes, parent_attractors = dynamic_values
        for attr in parent_attractors:
            for state in attr:
                yield state[:]
                i = 0
                while i != len(subgraph):
                    node = subgraph[i]
                    if state[node] + 1 < 2:
                        state[node] += 1
                        for j in range(i):
                            state[subgraph[j]] = 0
                        i = 0
                        yield state[:]
                    else:
                        i += 1


def transitions(net, size=None, subgraph=None, parent=None):
    if not is_network(net):
        raise TypeError("net is not a network")

    if is_fixed_sized(net):
        if size is not None:
            raise ValueError("size must be None for fixed sized networks")
        size = net.size
        state_space = net.state_space()
    else:
        if size is None:
            raise ValueError("size must not be None for variable sized networks")
        state_space = net.state_space(size)

    encoder = state_space._unsafe_encode;

    if subgraph is None:
        backward = None
        trans = [None] * state_space.volume;
        for i, state in enumerate(state_space):
            net._unsafe_update(state)
            trans[i] = (i, encoder(state))
    else:
        if parent is None:
            pin = [ n for n in range(state_space.ndim) if n not in subgraph ]
            trans = [None] * 2**len(subgraph)
            space = subspace(subgraph, size)
        else:
            parent_nodes, parent_attractors = parent
            pin = [ n for n in range(state_space.ndim) if n not in (subgraph | parent_nodes) ]
            trans = [None] * (sum(map(len, parent_attractors)) * 2**len(subgraph))
            space = subspace(subgraph, size, dynamic_values=parent)

        forward = {}
        backward = {}
        k = 0
        for i, state in enumerate(space):
            source = encoder(state)
            net._unsafe_update(state, pin=pin)
            target = encoder(state)

            if source not in forward:
                forward[source] = k
                backward[k] = source
                k += 1

            if target not in forward:
                forward[target] = k
                backward[k] = target
                k += 1

            trans[i] = (forward[source], forward[target])

    return backward, trans


def attractors_brute_force(net, size=None, subgraph=None, parent=None, encode=False):
    if not is_network(net):
        raise TypeError("net must be a network or a networkx DiGraph")
    elif is_fixed_sized(net) and size is not None:
        raise ValueError("fixed sized networks require size is None")
    elif not is_fixed_sized(net) and size is None:
        raise ValueError("variable sized networks require a size")

    if size is None:
        decoder = net.state_space().decode
    else:
        decoder = net.state_space(size).decode

    graph = nx.DiGraph()
    mapping, trans = transitions(net, size=size, subgraph=subgraph, parent=parent)
    graph.add_edges_from(trans)

    if mapping is None:
        if encode:
            attractors = list(nx.simple_cycles(graph))
        else:
            attractors = []
            for attr in nx.simple_cycles(graph):
                attractors.append(list(map(decoder, attr)))
    elif encode:
        attractors = []
        for attr in nx.simple_cycles(graph):
            attractors.append(list(map(lambda state: mapping[state], attr)))
    else:
        attractors = []
        for attr in nx.simple_cycles(graph):
            attractors.append(list(map(lambda state: decoder(mapping[state]), attr)))

    return attractors


def greatest_predecessors(dag, n):
    pred = list(dag.predecessors(n))
    N = len(pred)
    greatest = []
    for i in range(N):
        is_greatest = True
        for j in range(N):
            if i != j and nx.has_path(dag, pred[i], pred[j]):
                is_greatest = False
                break
        if is_greatest:
            greatest.append(pred[i])

    return greatest


def direct_sum(modules, attrs):
    def merge(nodes, attractor, component, matched_nodes=None):
        result = []
        for y in component:
            subresult = []
            for x in attractor:
                if matched_nodes is None or all([x[n] == y[n] for n in matched_nodes]):
                    state = y[:]
                    for i in nodes:
                        state[i] = x[i]
                    subresult.append(state)
            if len(subresult) != 0:
                result.append(subresult)
        return result

    if len(modules) == 0:
        return modules, attrs
    elif len(modules) == 1:
        return modules[0], attrs[0]
    else:
        attractors = []

        submodule, components = direct_sum(modules[1:], attrs[1:])
        matched_nodes = modules[0] & submodule
        for attractor in attrs[0]:
            subresult = []
            for component in components:
                subresult += merge(modules[0], attractor, component, matched_nodes)
            attractors += subresult
        submodule = modules[0] | submodule
        return submodule, attractors


def attractors(net, size=None, encode=True, max_module_size=15):
    if not is_network(net):
        raise TypeError("net must be a network or a networkx DiGraph")
    elif is_fixed_sized(net) and size is not None:
        raise ValueError("fixed sized networks require size is None")
    elif not is_fixed_sized(net) and size is None:
        raise ValueError("variable sized networks require a size")

    if size is None:
        g = net.to_networkx_graph()
        encoder = net.state_space()._unsafe_encode;
    else:
        g = net.to_networkx_graph(size)
        encoder = net.state_space(size)._unsafe_encode;

    modules = list(nx.strongly_connected_components(g))

    N = max(map(len, modules))
    if N> max_module_size:
        raise RuntimeError("largest module is too large ({} > {})".format(N, max_module_size))

    dag = nx.condensation(g)
    dag_list = list(nx.topological_sort(dag))

    attractors = {}

    for module_number in dag_list:
        parents = greatest_predecessors(dag, module_number)
        if len(parents) == 0:
            nodes = modules[module_number]
            attractors[module_number] = {
                        'eff_module': nodes,
                        'attractors': attractors_brute_force(net, size, subgraph=nodes)
                    }
        else:
            parent_modules = [ attractors[p]['eff_module'] for p in parents ]
            parent_attractors = [ attractors[p]['attractors'] for p in parents ]
            parent = direct_sum(parent_modules, parent_attractors)

            subgraph = modules[module_number]
            attractors[module_number] = {
                        'eff_module': parent[0] | subgraph,
                        'attractors': attractors_brute_force(net, size, subgraph=subgraph, parent=parent)
                    }

    outputs = list(filter(lambda m: len(list(dag.successors(m))) == 0, dag_list))
    parent_modules = [ attractors[o]['eff_module'] for o in outputs ]
    parent_attractors = [ attractors[o]['attractors'] for o in outputs ]

    _, attractors = direct_sum(parent_modules, parent_attractors)

    if encode:
        return list(map(lambda attractor: list(map(encoder, attractor)), attractors))
    else:
        return attractors

def main():
    import neet.synchronous as ns
    from neet.boolean.examples import s_pombe

    print(attractors(s_pombe, encode=True))

if __name__ == '__main__':
    main()
