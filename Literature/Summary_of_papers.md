# Literature Review - Short notes on papers found of interest - all papers are in the folder
October 8th 2019 - TD

**Seleson et al. 2009 - “PERIDYNAMICS AS AN UPSCALING OF MOLECULAR DYNAMICS”** - maths ‘multiscale’ paper including Max Gunsburger. Seeing Peridynamics as macroscale (higher-order) continuum model to a Molecular dynamic representation.  Paper is not that exciting really, and computations are limited to a simple diffusive example of little practical interest. Paper has good set of references to higher order or generalised mechanics literature.

**Kim et al. 2019 - “Peri-Net: Analysis of Crack Patterns Using Deep Neural Networks”** - Paper which throws a DNN to capture fracture patterns a train on Peridynamics simulation in LAMMPs. Paper claims they also solve the inverse problem, i.e. from a fracture pattern, predict the initial conditions (speed velocity of impact). Not clear how this is done from first read, since the inverse map seems very ill-posed (they don’t appear to use anything like a Invertible Neural Net either). Journal looks interesting - ‘Journal of Peridynamics and Nonlocal Modelling’

**Glaser et al. - “Strong scaling of general-purpose molecular dynamics simulations on GPUs”** Paper out of OakRidge on parallelisation using GPUs, gives information on how it is achieve in LAMMPs. 

**Parks et al. - "PDLAMMPS"** Original implementation details of PDLAMMPs - this what I followed for simple implementation, apart from the fact we are construct a gradient flow rather than second order system

