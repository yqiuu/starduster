.. starduster documentation master file, created by
   sphinx-quickstart on Sun Dec  5 11:56:30 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Starduster
==========
Starduster provides a deep learning model to emulate dust radiative transfer
simulations, which significantly accelerates the computation of dust
attenuation and emission. Starduster contains two specific generative models,
which explicitly take into accout the features of the dust attenuation curves
and dust emission spectra. Both generative models should be trained by a set of
characteristic outputs of a radiative transfer simulation. The obtained neural
networks can produce realistic galaxy spectral energy distributions that
satisfy the energy balance condition of dust attenuation and emission.
Applications of Starduster include SED-fitting and SED-modelling from semi-
analytic models. The code is written in PyTorch. Accordingly, users can take
advantage of GPU parallelisation and automatic differentiation implemented by
PyTorch throughout the applications.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   tutorials/sample_seds

.. toctree::
   :maxdepth: 2
   :caption: User guide

   starduster
   posterior
   analyzer

