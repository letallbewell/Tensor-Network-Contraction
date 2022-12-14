# Tensor-Network-Contraction
Program to contract arbitrary tensor networks. Currently, the contraction path has to be specified. We must incorporate optimal path-finding algorithms (a complex discrete optimization problem) or simplify the tensor network iteratively, sacrificing accuracy in favor of easier evaluation (renormalization methods) to contract more extensive networks.

## Introduction 

Tensor Networks are a new computational tool that finds applications in quantum many-body physics and machine learning. Among many of its applications, it allows for efficiently parametrizing high dimensional Hilbert spaces of quantum systems of interest and also provides a computationally tractable representation of partition functions involving lattice Hamiltonians (see [Ising Model](https://github.com/letallbewell/Ising_Model) for an application to the Ising Hamiltonian using the Google's Tensor Network library).

The network can be considered a graph representation of a complicated array contraction, like a long series of matrix multiplication. Since contraction is associative, the order does not make a difference to the answer, but the computational cost heavily relies on the order. Finding the optimal order is a discrete optimization problem(tough), and renormalization approaches can iteratively simplify the graph to give approximate answers.

## Ising Model Example

The following tensor network gives the partition function for the Ising Hamiltonian on a $2\times2$ lattice with periodic boundary conditions (larger networks failed to contract on my laptop).
![Ising TN](https://user-images.githubusercontent.com/43025445/195797650-01870fc9-f654-4d70-bf27-5bc90e25a242.jpg)

A naive contraction would look like this:
![Example Contraction](https://user-images.githubusercontent.com/43025445/195797826-6b87c70a-561a-4980-a3cb-b8371fe2e413.jpg)

We can see that the code works by checking the thermodynamic variables (these results are exact).

![Free energy](https://user-images.githubusercontent.com/43025445/195797978-b33029b5-ccb2-4761-9436-3c5bafd11698.jpg)
![TN E and C](https://user-images.githubusercontent.com/43025445/195798010-88da46f3-b666-4433-9df9-e355ce31c10a.jpg)

The dependence of computational cost on contraction order can be tested by contracting the edges in a few random orders:

![Contraction Times](https://user-images.githubusercontent.com/43025445/195798233-1090746d-3097-44ad-a0ec-d3dc05eb5888.jpg)

