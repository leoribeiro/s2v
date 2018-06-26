#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse, logging
import numpy as np
import struc2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time
import os
import graph
from utils import *

logging.basicConfig(filename='struc2vec.log',filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')

def parse_args():
	'''
	Parses the struc2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run struc2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--until-layer', type=int, default=None,
                    	help='Calculation until the layer.')

	parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=4,
	                    help='Number of parallel workers. Default is 4.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)
	return parser.parse_args()

def read_graph():
	'''
	Reads the input network.
	'''
	logging.info(" - Loading graph...")
	G = graph.load_edgelist(args.input,undirected=True)
	logging.info(" - Graph loaded.")
	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	logging.info("Initializing creation of the representations...")
	walks = [map(str, walk) for walk in walks]
	t0 = time()
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, hs=0, sg=1, workers=args.workers, iter=args.iter)
	t1 = time()
	print "Time skip-gram:",(t1-t0)
	model.save_word2vec_format(args.output)
	logging.info("Representations created.")
	
	return

def exec_struc2vec(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''

	#K = args.num_walks
	#S = args.walk_length 

	K = 10
	S = 5

	G = read_graph()
	G = struc2vec.Graph(G, args.directed, args.workers, K, S)

	G.preprocess_neighbors_with_rw()

	G.create_vectors()
	G.calc_distances()


	G.create_distances_network()

	walks = G.simulate_walks(args.num_walks, args.walk_length)

	return walks

def main(args):

	t0 = time()

	walks = exec_struc2vec(args)

	t1 = time()

	print "Time struc2vec:",(t1-t0)


	learn_embeddings(walks)


if __name__ == "__main__":
	args = parse_args()
	main(args)

