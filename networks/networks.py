#-*- coding: utf-8 -*-
from __future__ import (print_function, division, 
                        absolute_import, unicode_literals)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as colors

class Network(object):

    def __init__(self, nnodes, linklist=None, weighted=False,
                      directed=False):
        """Instance initialization method.

        Parameters
        ----------
            nnodes : int
                Number of nodes.

            linkslist : list of 2(3)-tuples 
                List with the connections between nodes. If the
                graphs is directed, a tuple (j, i) means that there
                is a link pointing from node j to node i. If the 
                graph is weighted, there is a third element in the
                tuple with the weight of the connection.

            weighted : bool
                If True, the graph is weighed.
                
            directed : bool
                If True, the connections in the graph have a direction.

        """
        # Store parameters
        self.nnodes = nnodes
        self.weighted = weighted
        self.directed = directed

        # Add the datatype of the adjacency matrix accoriding
        # to the type of graph
        if not self.weighted:
            self.dtype = bool
        else:
            self.dtype = float

        # Calculate and store the adjacency matrix 
        if linklist != None:
            self.adjmatrix = self._adjlist2matrix(
                    self.nnodes, linklist, self.weighted, self.directed)


    def neighbours_in(self, node):
        """Return an array with the nodes connected TO the given one.

        Returns the nodes j such that j -> node.
        In an undirected graph this method gives the same output 
        as neighbours_out.

        """
        inlinks = self.adjmatrix[:, node]
        return np.where(inlinks != 0)[0]  # The output is a tuple of arrays

    def neighbours_out(self, node):
        """Return an array with the nodes which the given one is connected TO.

        Returns the nodes j such that node -> j.
        In an undirected graph this method gives the same output 
        as neighbours_in.

        """
        outlinks = self.adjmatrix[node, :]
        return np.where(outlinks != 0)[0]  # The output is a tuple of arrays

    def update_link(self, link):
        """Set the new weight of the connection i -> j.

        This method is useful to handle undirected graphs, where i -> j 
        is the same as j -> i.

        """
        self.adjmatrix = self._setlink(adjmatrix, link, self.directed)
        return 


    @classmethod
    def _adjlist2matrix(cls, nnodes, linklist, weighted, directed):
        """Create the adjacency matrix from a list with the connections.

        Parameters
        ----------
            nnodes : int
                Number of nodes.

            linkslist : list of 2(3)-tuples 
                List with the connections between nodes. If the
                graphs is directed, a tuple (i, j) means that there
                is a link pointing from node i to node j. If the 
                graph is weighted, there is a third element in the
                tuple with the weight of the connection.

            weighted : bool
                If True, the graph is weighed.
                
            directed : bool
                If True, the connections in the graph have a direction.

        Returns
        -------
            adjmatrix : array
                (nnodes, nnodes) array where the element (i, j) 
                is the value of the link from i to j.

        """
        # Choose the data type according to type of graph
        if weighted:
            dtype = float
        else:
            dtype = bool
            # Add a weight ("True") to each link
            for link in linklist:
                link.append(True)

        # Initialize matrix
        adjmatrix = np.zeros((nnodes, nnodes), dtype=dtype)

        # Fill the matrix
        for link in linklist: 
            adjmatrix = cls._setlink(adjmatrix, link, directed)

        return adjmatrix


    @staticmethod
    def _setlink(adjmatrix, link, directed):
        """Set the new weight of the connection i -> j.

        This method is useful to handle undirected graphs, where i -> j 
        is the same as j -> i.

        Parameters
        ----------
            adjmatrix : 2d array
                Adjacency matrix to be updated.

            link : 2(3)-tuple 
                If the graphs is directed, a tuple (i, j) means that
                there is a link pointing from node i to node j (else it
                just means that there is an undirected link between the
                two). If the graph is weighted, there is a third element
                in the tuple with the weight of the connection.

            directed : bool
                If True, the connections in the graph have a direction.

        Returns
        -------
            adjmatrix : 2d array
                Updated adjacency matrix.

        """
        # Check if the weight of the link is given
        if len(link) == 3:
            newweight = link[2]
        else:
            newweight = 1

        # Update the link
        adjmatrix[link[0], link[1]] = newweight

        # If the graph is not directed, update the symmetric element
        # of the adjacency matrix
        if not directed:
            adjmatrix[link[1], link[0]] = newweight

        return adjmatrix
    
