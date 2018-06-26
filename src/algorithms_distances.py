# -*- coding: utf-8 -*-
from time import time
from collections import deque
import numpy as np
import math,logging
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from utils import *
import os

def getRWDegreeListsVertices(g,vertices,K,S):
    degreeList = {}

    for v in vertices:
        degreeList[v] = getRWDegreeLists(g,v,K,S)
        #degreeList[v] = getDegreeLists(g,v,S)

    return degreeList

def getDegreeLists(g, root, calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    

    l = deque()
    
    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        l.append(len(g[vertex]))

        for v in g[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1    

        if(timeToDepthIncrease == 0):

            lp = np.array(l,dtype='float')
            lp = np.sort(lp)
            listas[depth] = lp
            l = deque()

            if(calcUntilLayer - 1 == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root,(t1-t0)))


    return listas

def getRWDegreeLists(g,root,K,S):
    t0 = time()

    listas = {}
    paths = {}
    #print K,S
    
    for k in range(0,K):
        v = root
        path = deque()
        d = len(g[v])
        path.append(d)
        while len(path) < S:
            idx = np.random.randint(d)
            v = g[v][idx]
            d = len(g[v])
            path.append(d)
        paths[k] = path
    
    for s in range(0,S):
        l = []
        for k in range(0,K):
            l.append(paths[k][s])
        l = np.array(l,dtype='float')
        l = np.sort(l)
        listas[s] = np.array(l,dtype=np.int32)

    t1 = time()
    logging.info('RW vertex {}. Time: {}s'.format(root,(t1-t0)))

    return listas

def cost(a,b):
    ep = 0.5
    m = max(a,b) + ep
    mi = min(a,b) + ep
    return ((m/mi) - 1)

def cost_min(a,b):
    ep = 0.5
    m = max(a[0],b[0]) + ep
    mi = min(a[0],b[0]) + ep
    return ((m/mi) - 1) * min(a[1],b[1])


def cost_max(a,b):
    ep = 0.5
    m = max(a[0],b[0]) + ep
    mi = min(a[0],b[0]) + ep
    return ((m/mi) - 1) * max(a[1],b[1])

def verifyDegrees(degrees,degree_v_root,degree_a,degree_b):

    if(degree_b == -1):
        degree_now = degree_a
    elif(degree_a == -1):
        degree_now = degree_b
    elif(abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root)):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now 

def get_vertices(v,degree_v,degrees,a_vertices):
    a_vertices_selected = 2 * math.log(a_vertices,2)
    #logging.info("Selecionando {} próximos ao vértice {} ...".format(int(a_vertices_selected),v))
    vertices = deque()

    try:
        c_v = 0  

        for v2 in degrees[degree_v]['vertices']:
            if(v != v2):
                vertices.append(v2)
                c_v += 1
                if(c_v > a_vertices_selected):
                    raise StopIteration

        if('before' not in degrees[degree_v]):
            degree_b = -1
        else:
            degree_b = degrees[degree_v]['before']
        if('after' not in degrees[degree_v]):
            degree_a = -1
        else:
            degree_a = degrees[degree_v]['after']
        if(degree_b == -1 and degree_a == -1):
            raise StopIteration
        degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

        while True:
            for v2 in degrees[degree_now]['vertices']:
                if(v != v2):
                    vertices.append(v2)
                    c_v += 1
                    if(c_v > a_vertices_selected):
                        raise StopIteration

            if(degree_now == degree_b):
                if('before' not in degrees[degree_b]):
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]['before']
            else:
                if('after' not in degrees[degree_a]):
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]['after']
            
            if(degree_b == -1 and degree_a == -1):
                raise StopIteration

            degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

    except StopIteration:
        #logging.info("Vértice {} - próximos selecionados.".format(v))
        return list(vertices)

    return list(vertices)


def get_neighbors(list_vertices,G,degrees,number_vertices):

    vertices = {}
    
    for v in list_vertices:
        nbs = get_vertices(v,len(G[v]),degrees,number_vertices)
        vertices[v] = nbs

    return vertices


def calc_distances(vertices,degree_list,vertices_nbrs):

    distances = []

    dist_func = cost

    for v1 in vertices:
        #print v1,"->",
        lists_v1 = degree_list[v1]
        nbs = vertices_nbrs[v1]

        max_layer = len(degree_list[v1])

        for v2 in nbs:
            t00 = time()
            #print v2,
            #if(v1 == 11):
            #    print "v1",v1,"v2",v2," -> ",
            for layer in range(max_layer):
                #t00 = time()
                lists_v2 = degree_list[v2]
            
                dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)

                if(layer > 0):
                    f_dist = distances[-1]
                    dist = np.float32(dist + f_dist)
                #if(v1 == 11):
                #    print layer,dist,
                distances.append(dist)

            t11 = time()
            logging.info('fastDTW between vertices ({}, {}). Time: {}s'.format(v1,v2,(t11-t00)))
            #if(v1 == 11):
            #    print "#"
    return distances


def selectVertices(layer,fractionCalcDists):
    previousLayer = layer - 1

    logging.info("Recovering distances from disk...")
    distances = restoreVariableFromDisk('distances')

    threshold = calcThresholdDistance(previousLayer,distances,fractionCalcDists)

    logging.info('Selecting vertices...')

    vertices_selected = deque()

    for vertices,layers in distances.iteritems():
        if(previousLayer not in layers):
            continue
        if(layers[previousLayer] <= threshold):
            vertices_selected.append(vertices)

    distances = {}

    logging.info('Vertices selected.')

    return vertices_selected


def exec_rw(G,workers,K,S):

    futures = {}
    degreeList = {}

    t0 = time()
    vertices = G.keys()
    parts = workers
    chunks = partition(vertices,parts)

    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            #dl = getRWDegreeListsVertices(G,c,K,S)
            #degreeList.update(dl)
            job = executor.submit(getRWDegreeListsVertices,G,c,K,S)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList.update(dl)

    #logging.info("Saving degreeList on disk...")
    #saveVariableOnDisk(degreeList,'degreeList')
    t1 = time()
    logging.info('Execution time - RW BFS: {}m'.format((t1-t0)/60))


    return degreeList


def generate_parameters_random_walk(multi_graph):

    sum_weights = {}
    amount_edges = {}

    for v in multi_graph.keys():

        max_layer = multi_graph[v].shape[0] - 1
        for layer,weights in enumerate(multi_graph[v]):

            if(layer >= max_layer):
                break

            if(layer not in sum_weights):
                sum_weights[layer] = 0
                amount_edges[layer] = 0
            
            sum_weights[layer] += np.sum(weights)
            amount_edges[layer] += len(weights)

    average_weight = {}
    for layer in sum_weights.keys():
        average_weight[layer] = sum_weights[layer] / amount_edges[layer]


    amount_neighbours = {}
    for v in multi_graph.keys():

        max_layer = multi_graph[v].shape[0] - 1
        for layer,weights in enumerate(multi_graph[v]):

            if(layer >= max_layer):
                break

            if(layer not in amount_neighbours):
                amount_neighbours[layer] = {}

            amount_neighbours[layer][v] = 0

            for w in weights:
                if(w > average_weight[layer]):
                    amount_neighbours[layer][v] += 1

    return amount_neighbours

def generate_multi_graph(all_distances,chunks_vertices,vertices_nbrs,max_layer):

    multi_graph = {}

    for index,vertices in enumerate(chunks_vertices):    
        distances = all_distances[index]
        cont_distance = 0
        #print vertices,index

        for v1 in vertices:
            
            #print v1,"->",
            nbs = vertices_nbrs[v1]
            multi_graph[v1] = np.full((max_layer+1,len(nbs)), np.inf)

            multi_graph[v1][max_layer] = np.array(nbs)

            for idx,v2 in enumerate(nbs):
                #if(v1 == 13 or v1 == 14):
                #    print "v1",v1,"v2",v2," ->",
                for layer in range(max_layer):
                    dist = distances[cont_distance]
                    #if(v1 == 14 or v1 == 13):
                    #    print layer,dist," ",
                    w = np.exp(-float(dist))
                    multi_graph[v1][layer][idx] = w
                    cont_distance += 1
                #if(v1 == 13 or v1 == 14):
                #    print " #"
        


            #print "#"


    return multi_graph


def generate_multigraph_probabilities(multi_graph):

    alias_method_j = {}
    alias_method_q = {} 


    for v in multi_graph.keys():

        max_layer = multi_graph[v].shape[0] - 1
        len_nbs = multi_graph[v].shape[1]

        alias_method_j[v] = np.full((max_layer,len_nbs), np.inf)
        alias_method_q[v] = np.full((max_layer,len_nbs), np.inf)

        for layer,weights in enumerate(multi_graph[v]):

            if(layer >= max_layer):
                break

            unnormalized_probs = multi_graph[v][layer]
            norm_const = sum(unnormalized_probs)

            multi_graph[v][layer] = \
            np.array([float(u_prob)/norm_const for u_prob in unnormalized_probs])

            J, q = alias_setup(multi_graph[v][layer])
            alias_method_j[v][layer] = J
            alias_method_q[v][layer] = q

    return multi_graph,alias_method_j,alias_method_q


def generate_distances_network(all_distances,vertices,vertices_nbrs,max_layer):
    t0 = time()
    logging.info('Creating distance network...')

    multi_graph = generate_multi_graph(all_distances,vertices,vertices_nbrs,max_layer)

    t1 = time()
    t = t1-t0
    logging.info('- Time - part 1: {}s'.format(t))


    amount_neighbours = generate_parameters_random_walk(multi_graph)

    multi_graph,alias_method_j,alias_method_q = generate_multigraph_probabilities(multi_graph)
 
    return multi_graph,alias_method_j,alias_method_q,amount_neighbours


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q
