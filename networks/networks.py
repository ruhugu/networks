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

        # Create the adjacency matrix
        self.adjmatrix = np.zeros((self.nnodes, self.nnodes), dtype=self.dtype)

#        # Calculate and store the adjacency matrix 
#        if linklist != None:
#            self.adjmatrix = self._adjlist2matrix(
#                    self.nnodes, linklist, self.weighted, self.directed)

    # Network info
    # ========================================
    def neighbours_in(self, node):
        """Return an array with the nodes pointing TO the given one.

        Returns the nodes j such that j -> node.
        In an undirected graph this method gives the same output 
        as neighbours_out.

        """
        inlinks = self.adjmatrix[:, node]
        return np.where(inlinks != 0)[0]  # The output is a tuple of arrays

    def neighbours_out(self, node):
        """Return an array with the nodes which the given one is pointing TO.

        Returns the nodes j such that node -> j.
        In an undirected graph this method gives the same output 
        as neighbours_in.

        """
        outlinks = self.adjmatrix[node, :]
        return np.where(outlinks != 0)[0]  # The output is a tuple of arrays

    @property
    def degrees_in(self):
        """Return 1D array with the "in" degree of each node.

        """
        return np.sum(self.adjmatrix.astype(bool), axis=0)

    @property
    def degrees_out(self):
        """Return 1D array with the "out" degree of each node.

        """
        return np.sum(self.adjmatrix.astype(bool), axis=1)

    def adjlist(self, weighted=None, directed=None):
        """Return the adjacency list of the graph.

        """
        if weighted == None:
            weighted = self.weighted

        if directed == None:
            directed = self.directed

        return self.adjmatrix2list(
                self.adjmatrix, weighted=weighted, directed=directed)


    # Network manipulation
    # ========================================
    def update_link(self, link, newweight=1):
        """Set the new weight of the connection i -> j.

        This method is useful to handle undirected graphs, where i -> j 
        is the same as j -> i.

        Parameters
        ----------
            link : 2-tuple 
                If the graphs is directed, a tuple (i, j) refers to the
                 link pointing from node i to node j (else it
                just means that there is an undirected link between the
                two).
            newweight : float
                New value of the weight. If the network is not weighted
                any value different from zero creates a new link while
                a value of zero removes it.
                
        """
        # TODO: Improve this, right now it looks ugly
        self.adjmatrix = self._setlink(
                self.adjmatrix, link + [newweight,], self.directed)
        return 

    def remove_link(self, link):
        """Remove the link.

        Parameters
        ----------
            link : 2-tuple 
                If the graphs is directed, a tuple (i, j) refers to the
                link pointing from node i to node j (else it just means
                that there is an undirected link between the two).

        """
        self.update_link(link, newweight=0)
        return 

    def read_adjlist(self, linklist):
        """Add (or update) links to graph from a list.

        Parameters
        ----------
            linklist : list of 2(3)-tuples 
                List with the connections between nodes. If the
                graphs is directed, a tuple (i, j) means that there
                is a link pointing from node i to node j. If the 
                graph is weighted, there is a third element in the
                tuple with the weight of the connection.

        """
        self.adjmatrix = self.adjlist2matrix(
                self.nnodes, linklist, weighted=self.weighted,
                directed=self.directed)
        return

    # Auxiliar functions
    # ========================================
    @classmethod
    def adjlist2matrix(cls, nnodes, linklist, weighted, directed):
        """Create the adjacency matrix from a list with the connections.

        Parameters
        ----------
            nnodes : int
                Number of nodes.

            linklist : list of 2(3)-tuples 
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

        # Initialize matrix
        adjmatrix = np.zeros((nnodes, nnodes), dtype=dtype)

        # Fill the matrix
        for link in linklist: 
            adjmatrix = cls._setlink(adjmatrix, link, directed)

        return adjmatrix


    @classmethod
    def adjmatrix2list(cls, adjmatrix, weighted=True, directed=True):
        """Create an adjacency list from the adjacency matrix.

        Parameters
        ----------
            adjmatrix : int or bool 2D array
                Adjacency matrix.

            weighted : bool
                If True, the weigth of the links are stored in the list.
                
            directed : bool
                If False, only the links in the lower side of the 
                adjacency matrix are store (since the matrix is 
                symmetric).

        Returns
        -------
            adjlist : 2(3)-tuple list
                List with the connections between nodes. If the
                graphs is directed, a tuple (i, j) means that there
                is a link pointing from node i to node j. If the 
                graph is weighted, there is a third element in the
                tuple with the weight of the connection.

        """
        # If the graph is not directed, ignore the upper side of
        # the adjacency matrix
        if not directed:
            auxmatrix = np.tril(adjmatrix)
        else:
            auxmatrix = adjmatrix

        # Initialize list
        adjlist = list()

        for j_node in range(auxmatrix.shape[0]):
            # Find the nodes which j_node is pointing to
            outlinks = auxmatrix[j_node, :]
            neighs_out = np.where(outlinks != 0)[0]  

            # Store the links in the list
            for neigh in neighs_out:
                link = [j_node, neigh]
                if weighted:
                    link.append(float(auxmatrix[j_node, neigh]))

                adjlist.append(link)

        return adjlist


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

    
# TODO: this should be removed and replaced with Lattice
#class Lattice2D(Network):
#    """Regular 2D lattice network.
#
#    """
#    def __init__(self, nrows, ncols, pbc=True, weighted=False, directed=False):
#        """Instance initialization method.
#
#        Parameters
#        ----------
#            nrows : int
#                Number rows in the 2D lattice.
#
#            ncols : int
#                Number columns in the 2D lattice.
#            
#            pbc : bool
#                If True, periodic boundary conditions are used. 
#
#        """
#        # Store parameters
#        self.nrows = nrows
#        self.ncols = ncols
#        self.pbc = pbc
#
#        self.nnodes = self.nrows*self.ncols
#
#        Network.__init__(
#                self, self.nnodes, weighted=weighted, directed=directed)
#
#        # Calculate the adjacency list and update the network
#        adjlist = regularlattice_list((nrows, ncols), pbc=self.pbc)
#        self.read_adjlist(adjlist)


class Lattice(Network):
    """Regular N-dimensional lattice network.

    A node with index (j0, j1, .., jm) with m the size of shape is
    numbered according to the following rule:
        j_node = j0 + j1*shape[0] + j2*shape[0]*shape[1] 
                + ... + jm*np.prod(shape[0:m])

    """
    def __init__(self, shape, pbc=True, weighted=False, directed=False):
        """Instance initialization method.

        Parameters
        ----------
            shape : int tuple
                Shape of the lattice. 

            pbc : bool
                If True, periodic boundary conditions are used. 

            weighted : bool
                If True, the weigth of the links are stored in the list.
                
            directed : bool
                If False, only the links in the lower side of the 
                adjacency matrix are store (since the matrix is 
                symmetric).

        """
        # Store parameters
        self.shape = shape
        self.pbc = pbc
        # (weighted and directed parameters are stored when calling 
        # Network's __init__ method)

        # Calculate the number of nodes in the Network
        self.nnodes = np.prod(self.shape)

        Network.__init__(
                self, self.nnodes, weighted=weighted, directed=directed)

        # Calculate the adjacency list and update the network
        adjlist = self.regularlattice_list(self.shape, pbc=self.pbc)
        self.read_adjlist(adjlist)


    @staticmethod
    def regularlattice_list(shape, pbc=True):
        """Return the adjacency list of a regular lattice network.

        A node with index (j0, j1, .., jm) with m the size of shape is
        numbered according to the following rule:
            j_node = j0 + j1*shape[0] + j2*shape[0]*shape[1] 
                    + ... + jm*np.prod(shape[0:m])
        Only the nearest neighbours are included in the list.

        Parameters
        ----------
            shape : int or sequence of ints
                Shape of the network. For example, a square network with 
                side 3 would have shape (3, 3).

            Returns
            -------
                adjlist : 2-tuple list
                    List with the connections between nodes. A tuple (i, j)
                    means that there is a link pointing from node i to node j.

        """
        # Calculate the number of nodes in the network
        nnodes = np.product(shape)
        dim = len(shape)

        adjlist = list()

        for j_node in range(nnodes):
            # Find the index of the j-th node in the network
            idx = np.unravel_index(j_node, shape)

            # Find the adjacent nodes in each direction
            for j_dim in range(dim):
                vec = np.zeros(dim, dtype=int)
                vec[j_dim] += 1

                # Node behind j_node in the j_dim direction
                if idx[j_dim] != 0 or pbc:
                    j_neigh = np.ravel_multi_index(
                            (idx - vec), shape, mode="wrap") 
                    adjlist.append([j_node, j_neigh])

                # Node in front of j_node in the j_dim direction
                if idx[j_dim] != (shape[j_dim] - 1) or pbc:
                    j_neigh = np.ravel_multi_index(
                            (idx + vec), shape, mode="wrap") 
                    adjlist.append([j_node, j_neigh])

        return adjlist
        self.read_adjlist(adjlist)


def regularnetwork_list(shape, pbc=True):
    """Return the adjacency list of a regular network.

    Only the nearest neighbours are included in the list.

    Parameters
    ----------
        shape : int or sequence of ints
            Shape of the network. For example, a square network with 
            side 3 would have shape (3, 3).

        Returns
        -------
            adjlist : 2-tuple list
                List with the connections between nodes. A tuple (i, j)
                means that there is a link pointing from node i to node j.

    """
