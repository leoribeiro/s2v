# -*- coding: utf-8 -*-

import numpy as np
import random,sys,logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from time import time
from collections import deque

from utils import *
from algorithms import *
from algorithms_distances import *
import graph


class Graph():
	def __init__(self, g, is_directed, workers, K, S):

		logging.info(" - Converting graph to dict...")
		self.G = g.gToDict()
		logging.info("Graph converted.")

		self.num_vertices = g.number_of_nodes()
		self.num_edges = g.number_of_edges()
		self.is_directed = is_directed
		self.workers = workers
		self.K = K
		self.S = S
		logging.info('Graph - Number of vertices: {}'.format(self.num_vertices))
		logging.info('Graph - Number of edges: {}'.format(self.num_edges))


	def preprocess_neighbors_with_rw(self):

		# with ProcessPoolExecutor(max_workers=self.workers) as executor:
		# 	job = executor.submit(exec_rw,self.G,self.workers,self.K,self.S)
			
		# 	self.degree_list = job.result()

		self.degree_list = exec_rw(self.G,self.workers,self.K,self.S)

		return

	def create_vectors(self):
		logging.info("Creating degree vectors...")
		degrees = {}
		degrees_sorted = set()
		G = self.G
		for v in G.keys():
			degree = len(G[v])
			degrees_sorted.add(degree)
			if(degree not in degrees):
				degrees[degree] = {}
				degrees[degree]['vertices'] = deque() 
			degrees[degree]['vertices'].append(v)
		degrees_sorted = np.array(list(degrees_sorted),dtype='int')
		degrees_sorted = np.sort(degrees_sorted)

		l = len(degrees_sorted)
		for index, degree in enumerate(degrees_sorted):
			if(index > 0):
				degrees[degree]['before'] = degrees_sorted[index - 1]
			if(index < (l - 1)):
				degrees[degree]['after'] = degrees_sorted[index + 1]
		logging.info("Degree vectors created.")

		self.degrees_vector = degrees

	def calc_distances(self):

		futures = {}
		results = {}

		G = self.G
		number_vertices = len(G)

		vertices_ = G.keys()
		vertices_nbrs = get_neighbors(vertices_,G,self.degrees_vector,number_vertices)
		chunks_vertices = partition(vertices_,self.workers)

		distances = {}

		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 0
			for c in chunks_vertices:
				#print c,part
				#print "part",part
				#calc_distances(c, self.degree_list, vertices_nbrs)
				
				logging.info("Executing part {}...".format(part))
				job = executor.submit(calc_distances, c, self.degree_list, vertices_nbrs)
				futures[job] = part
				part += 1

			logging.info("Receiving results...")
			for job in as_completed(futures):
				part = futures[job]
				distances[part] = job.result()
				logging.info("Part {} completed.".format(part))



		self.distances = distances
		self.vertices_nbrs = vertices_nbrs
		return


	def create_distances_network(self):
		chunks_vertices = partition(self.G.keys(),self.workers)

		self.multi_graph,self.alias_method_j,self.alias_method_q,self.amount_neighbours = \
		generate_distances_network(self.distances,chunks_vertices,self.vertices_nbrs,self.S)

		return


	def simulate_walks(self,num_walks,walk_length):

		return generate_random_walks(num_walks,walk_length,self.workers,
			self.G.keys(),self.multi_graph,
			self.alias_method_j,self.alias_method_q,self.amount_neighbours)

		# for large graphs, it is serially executed, because of memory use.
		# if(len(self.G) > 500000):

		# 	with ProcessPoolExecutor(max_workers=1) as executor:
		# 		job = executor.submit(generate_random_walks_large_graphs,num_walks,walk_length,self.workers,self.G.keys())

		# 		job.result()

		# else:

		# 	with ProcessPoolExecutor(max_workers=1) as executor:
		# 		job = executor.submit(generate_random_walks,num_walks,walk_length,self.workers,self.G.keys())

		# 		job.result()


			





		

      	


