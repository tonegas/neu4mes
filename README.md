# Neu4mes 
[![License MIT](https://go-shields.herokuapp.com/license-MIT-blue.png)]()  [![Travis-CI Status](https://app.travis-ci.com/tonegas/neu4mes.svg?branch=master)](https://travis-ci.org/tonegas/neu4mes)  [![Coverage Status](https://coveralls.io/repos/github/tonegas/neu4mes/badge.svg?branch=master)](https://coveralls.io/github/tonegas/neu4mes?branch=master)
## Mechanics-informed neural network framework for modeling and control mechanical system
Mechanics-informed neural networks (MINNs) are a new neural networks concept. These networks base their structure on physical (mainly mechanics) and control theory laws. 

The framework's goal is to allow the users fast modeling and control of a mechanical system such as an autonomous vehicle, an industrial robot, a walking robot, a flying drone.

A conceptual representation of the mechanical system will be used to obtain a neural network of the MINN type, which has an optimal structure to model the considered mechanical device. 
The framework will realize the training by appropriately choosing all the hyper-parameters and allowing adequate training by providing suitable experimental data.
The framework will allow the user to synthesize and train a neural network that will be used as a control system in a few simple steps and without the need to perform new experiments. 
The realized controller will be exported independently from any external library in the C language.
