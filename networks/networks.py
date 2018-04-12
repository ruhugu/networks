#-*- coding: utf-8 -*-
from __future__ import (print_function, division, 
                        absolute_import, unicode_literals)
import numpy as np

class Network(object):

    def __init__(self, nnodes, edgelist=None, weighted=False,
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
#        if edgelist != None:
#            self.adjmatrix = self._adjlist2matrix(
#                    self.nnodes, edgelist, self.weighted, self.directed)

    # Network info
    # ========================================
    def neighbours_in(self, node):
        """Return an array with the nodes pointing TO the given one.

        Returns the nodes j such that j -> node.
        In an undirected graph this method gives the same output 
        as neighbours_out.

        """
        inedges = self.adjmatrix[:, node]
        return np.where(inedges != 0)[0]  # The output is a tuple of arrays

    def neighbours_out(self, node):
        """Return an array with the nodes which the given one is pointing TO.

        Returns the nodes j such that node -> j.
        In an undirected graph this method gives the same output 
        as neighbours_in.

        """
        outedges = self.adjmatrix[node, :]
        return np.where(outedges != 0)[0]  # The output is a tuple of arrays

    @property
    def degree_in(self):
        """Return 1D array with the "in" degree of each node.

        """
        return np.sum(self.adjmatrix.astype(bool), axis=0)

    @property
    def degree_out(self):
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
    def update_edge(self, edge, newweight=1):
        """Set the new weight of the connection i -> j.

        This method is useful to handle undirected graphs, where i -> j 
        is the same as j -> i.

        Parameters
        ----------
            edge : 2-tuple 
                If the graphs is directed, a tuple (i, j) refers to the
                 edge pointing from node i to node j (else it
                just means that there is an undirected edge between the
                two).
            newweight : float
                New value of the weight. If the network is not weighted
                any value different from zero creates a new edge while
                a value of zero removes it.
                
        """
        # TODO: Improve this, right now it looks ugly
        self.adjmatrix = self._setedge(
                self.adjmatrix, edge + [newweight,], self.directed)
        return 

    def remove_edge(self, edge):
        """Remove the edge.

        Parameters
        ----------
            edge : 2-tuple 
                If the graphs is directed, a tuple (i, j) refers to the
                edge pointing from node i to node j (else it just means
                that there is an undirected edge between the two).

        """
        self.update_edge(edge, newweight=0)
        return 

    def read_adjlist(self, edgelist):
        """Add (or update) edges to graph from a list.

        Parameters
        ----------
            edgelist : list of 2(3)-tuples 
                List with the connections between nodes. If the
                graphs is directed, a tuple (i, j) means that there
                is a edge pointing from node i to node j. If the 
                graph is weighted, there is a third element in the
                tuple with the weight of the connection.

        """
        self.adjmatrix = self.adjlist2matrix(
                self.nnodes, edgelist, weighted=self.weighted,
                directed=self.directed)
        return


    # Network properties
    # ========================================
    def clusteringcoeff(self, node):
        """Calculate the clustering coefficient of the given node. 

        This method only works for undirected networks.

        See Wikipedia page:
        https://en.wikipedia.org/wiki/Clustering_coefficient

        """
        if self.directed:
            raise ValueError(
                    "This method does not work with directed networks.")

        # Find the neighbours of the node and its degree
        neighs = self.neighbours_out(node)
        degree = self.degree_out[node]

        # Calculate the number of conections between neighbours
        neighpairs = np.sum(np.triu(self.adjmatrix[neighs][:, neighs]))

        # Number of possible pairs
        maxpairs = degree*(degree - 1)/2

        return float(neighpairs)/maxpairs

    def clusteringcoeff_mean(self):
        """Calculate the mean clustering coefficient of the network.

        """
        return np.mean(
                [self.clusteringcoeff(node) for node in range(self.nnodes)])

    def degree_out_dist(self):
        """Return the out degree distribution of the network.
        
        """
        return np.bincount(self.degree_put).astype(float)/self.nnodes
    
    def degree_in_dist(self):
        """Return the in degree distribution of the network.
        
        """
        return np.bincount(self.degree_in).astype(float)/self.nnodes

    def distance(self, i_node, j_node):
        """Return the shortest path length between two nodes.

        Parameters
            i_node : int
                Start node.

            j_node : int 
                End node.

        """

    # Auxiliar functions
    # ========================================
    @classmethod
    def adjlist2matrix(cls, nnodes, edgelist, weighted, directed):
        """Create the adjacency matrix from a list with the connections.

        Parameters
        ----------
            nnodes : int
                Number of nodes.

            edgelist : list of 2(3)-tuples 
                List with the connections between nodes. If the
                graphs is directed, a tuple (i, j) means that there
                is a edge pointing from node i to node j. If the 
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
                is the value of the edge from i to j.

        """
        # Choose the data type according to type of graph
        if weighted:
            dtype = float
        else:
            dtype = bool

        # Initialize matrix
        adjmatrix = np.zeros((nnodes, nnodes), dtype=dtype)

        # Fill the matrix
        for edge in edgelist: 
            adjmatrix = cls._setedge(adjmatrix, edge, directed)

        return adjmatrix


    @classmethod
    def adjmatrix2list(cls, adjmatrix, weighted=True, directed=True):
        """Create an adjacency list from the adjacency matrix.

        Parameters
        ----------
            adjmatrix : int or bool 2D array
                Adjacency matrix.

            weighted : bool
                If True, the weigth of the edges are stored in the list.
                
            directed : bool
                If False, only the edges in the lower side of the 
                adjacency matrix are store (since the matrix is 
                symmetric).

        Returns
        -------
            adjlist : 2(3)-tuple list
                List with the connections between nodes. If the
                graphs is directed, a tuple (i, j) means that there
                is a edge pointing from node i to node j. If the 
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
            outedges = auxmatrix[j_node, :]
            neighs_out = np.where(outedges != 0)[0]  

            # Store the edges in the list
            for neigh in neighs_out:
                edge = [j_node, neigh]
                if weighted:
                    edge.append(float(auxmatrix[j_node, neigh]))

                adjlist.append(edge)

        return adjlist


    @staticmethod
    def _setedge(adjmatrix, edge, directed):
        """Set the new weight of the connection i -> j.

        This method is useful to handle undirected graphs, where i -> j 
        is the same as j -> i.

        Parameters
        ----------
            adjmatrix : 2d array
                Adjacency matrix to be updated.

            edge : 2(3)-tuple 
                If the graphs is directed, a tuple (i, j) means that
                there is a edge pointing from node i to node j (else it
                just means that there is an undirected edge between the
                two). If the graph is weighted, there is a third element
                in the tuple with the weight of the connection.

            directed : bool
                If True, the connections in the graph have a direction.

        Returns
        -------
            adjmatrix : 2d array
                Updated adjacency matrix.

        """
        # Check if the weight of the edge is given
        if len(edge) == 3:
            newweight = edge[2]
        else:
            newweight = 1

        # Update the edge
        adjmatrix[edge[0], edge[1]] = newweight

        # If the graph is not directed, update the symmetric element
        # of the adjacency matrix
        if not directed:
            adjmatrix[edge[1], edge[0]] = newweight

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
                If True, the weigth of the edges are stored in the list.
                
            directed : bool
                If False, only the edges in the lower side of the 
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
                    means that there is a edge pointing from node i to node j.

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


class Regular(Network):
    """Regular network. 
    
    The resulting network is unweighted and undirected.

    See Wikipedia page:
    https://en.wikipedia.org/wiki/Regular_graph

    """
    def __init__(self, nnodes, grade):
        """Instance initialization method.

        Parameters
        ----------
            nnodes : int
                Number of nodes in the lattice.

            grade : bool
                Number of neighbours of each node. It must be a even
                number, so that any node has the same number of 
                neighbours at both sides.

        """
        # Store parameters
        self.grade = grade
        self.nnodes = nnodes

        Network.__init__(
                self, nnodes, weighted=False, directed=False)

        # Calculate the adjacency list and update the network
        adjlist = self.regular_list(nnodes, grade)
        self.read_adjlist(adjlist)


    @staticmethod
    def regular_list(nnodes, grade):
        """Return the adjacency list of a regular network.

        Parameters
        ----------
            nnodes : int
                Number of nodes in the network.

            grade : int
                Number of neighbours of each node. It must be a even
                number, so that any node has the same number of 
                neighbours at both sides.

            Returns
            -------
                adjlist : 2-tuple list
                    List with the connections between nodes. A tuple (i, j)
                    means that there is a edge pointing from node i to node j.
                        
        """
        # If grade is not even raise error
        if (grade % 2 != 0):
            raise ValueError("grade must be an even number.")

        # Initialize adjacency list
        adjlist = list()

        for j_node in range(nnodes):
            for dist in range(1, grade//2 + 1):
                adjlist.append([j_node, (j_node + dist) % nnodes])
                adjlist.append([j_node, (j_node - dist) % nnodes])

        return adjlist


class WattsStrogatz(Regular):
    """Watts-Strogatz network.

    See Wikipedia page:
    https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model

    """
