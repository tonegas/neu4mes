# Neu4mes
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  [![Travis-CI Status](https://app.travis-ci.com/tonegas/neu4mes.svg?branch=master)](https://travis-ci.org/tonegas/neu4mes)  [![Coverage Status](https://coveralls.io/repos/github/tonegas/neu4mes/badge.svg?branch=master)](https://coveralls.io/github/tonegas/neu4mes?branch=master)
<a name="readme-top"></a>
## Structured neural network framework for modeling and control mechanical system
_Structured neural networks_ (SNNs) are a new neural networks concept. 
These networks base their structure on mechanical and control theory laws. 

The framework's goal is to allow the users fast modeling and control of a mechanical system such as an autonomous vehicle, an industrial robot, a walking robot, a flying drone.

Below is the workflow that the framework follows.

Using a conceptual representation of your mechanical system the framework generates the structured neural network of model of mechanical device considered. 
Providing suitable experimental data, the framework will realize an effective training of the neural models by appropriately choosing all the hyper-parameters.
The framework will allow the user to synthesize and train a structured neural network that will be used as a control system in a few simple steps and without the need to perform new experiments. 
The realized neural controller will be exported using C language or ONNX, and it will be ready to use.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#settingstarted">Getting Started</a>
    </li>
    <li>
      <a href="#basicfunctionalities">Basic Functionalities</a>
      <ul>
        <li><a href="#structuredneuralmodel">Build the structured neural model</a></li>
        <li><a href="#neuralizemodel">Neuralize the structured neural model</a></li>
        <li><a href="#loaddataset">Load the dataset</a></li>
        <li><a href="#trainmodel">Train the structured neural network</a></li>
        <li><a href="#testmodel">Test the structured neural model</a></li>
      </ul>
    </li>
    <li>
      <a href="#fonlderstructure">Structure of the Folders</a>
      <ul>
        <li><a href="#neu4mesfolder">neu4mes folder</a></li>
        <li><a href="#examplesfolder">examples folder</a></li>
        <li><a href="#testsfolder">tests folder</a></li>
        <li><a href="#underdevfolder">underdev folder</a></li>
        <li><a href="#usagefolder">usage folder</a></li>
      </ul>
    </li>
    <li>
      <a href="#shape">Overview on Signal Shape</a>
      <ul>
        <li><a href="#inoutshape">Input and output shape from the structured neural model</a></li>
        <li><a href="#elementwiseshape">Elementwise Arithmetic, Activation, Trigonometric</a></li>
        <li><a href="#firshape">Fir</a></li>
        <li><a href="#linearshape">Linear</a></li>
        <li><a href="#fuzzyshape">Fuzzy</a></li>
        <li><a href="#partshape">Part and Select</a></li>
        <li><a href="#timepartshape">TimePart, SimplePart, SampleSelect</a></li>
        <li><a href="#localmodelshape">LocalModel</a></li>
        <li><a href="#parametersshape">Parameters</a></li>
        <li><a href="#paramfunshape">Parametric Function</a></li>
      </ul>
    </li>
  </ol>
</details>

<!-- GETTING STARTED -->
<a name="settingstarted"></a>
## Getting Started
### Prerequisites

You can install the dependencies of the neu4mes framework from PyPI via:
  ```sh
  pip install -r requirements.txt
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="basicfunctionalities"></a>
## Basic Functionalities
<a name="structuredneuralmodel"></a>
### Build the structured neural model

The structured neural model is defined by a list of inputs by a list of outputs and by a list of relationships that link the inputs to the outputs.

Let's assume we want to model one of the best-known linear mechanical systems, the mass-spring-damper system.

<p align="center">
<img src="img/massspringdamper.png" width="250" alt="linearsys" >
</p>

The system is defined as the following equation:
```math
M \ddot x  = - k x - c \dot x + F
```

Suppose we want to estimate the value of the future position of the mass given the initial position and the external force.

In the neu4mes framework we can build an estimator in this form:
```python
x = Input('x')
F = Input('F')
x_z = Output('x_z_est', Fir(x.tw(1))+Fir(F.last()))
```

The first thing we define the input variable of the system.
Input variabiles can be created using the `Input` function.
In our system we have two inputs the position of the mass, `x`, and the external force, `F`, exerted on the mass.
The `Output` function is used to define an output of our model.
The `Output` gets two inputs, the first is the name of the output and the second is the structure of the estimator.

Let's explain some of the functions used:
1. The `tw(...)` function is used to extract a time window from a signal. 
In particular we extract a time window of 1 second.
2. The `last()` function that is used to get the last force applied to the mass.
3. The `Fir(...)` function to build an FIR filter with the tunable parameters on our input variable.

So we are creating an estimator for the variable `x` at the instant following the observation (the future position of the mass) by building an 
observer that has a mathematical structure equal to the one shown below:
```math
x[1] = \sum_{k=0}^{N_x-1} x[-k]\cdot h_x[(N_x-1)-k] + F[0]\cdot h_F
```
Where the variables $N_x$, and $h_f$ also the values of the vectors $h_x$ are still unknowns.
Regarding $N_x$, we know that the window lasts one second but we do not know how many samples it corresponds to and this depends on the discretization interval.
The formulation above is equivalent to the formulation of the discrete time response of the system
if we choose $N_x = 3$ and $h_x$ equal to the characteristic polynomial and $h_f = T^2/M$ (with $T$ sample time).
Our formulation is more general and can take into account the noise of the measured variable using a bigger time window.
The estimator can also be seen as the composition of the force contributions due to the position and velocity of the mass plus the contribution of external forces.

<a name="neuralizemodel"></a>
### Neuralize the structured neural model
Let's now try to train our observer using the data we have.
We perform:
```python
mass_spring_damper = Neu4mes()
mass_spring_damper.addModel(x_z_est)
mass_spring_damper.minimizeError('next-pos', x.z(-1), x_z_est, 'mse')
mass_spring_damper.neuralizeModel(0.2)
```
Let's create a **neu4mes** object, and add one output the network using the `addModel` function.
This function is needed for create an output on the model. In this example it is not mandatory because the same output is added also to the `minimizeError` function.
In order to train our model/estimator the function `minimizeError` is used to add a loss function to the list of losses.
This function takes:
1. The name of the error, it is presented in the results and during the training.
2. The second and third inputs are the variable that will be minimized, the order is not important.
3. The minimization function used, in  this case 'mse'.
In the function `minimizeError` is used the `z(-1)` function. This function get from the dataset the future value of a variable 
(in our case the position of the mass), the next instant, using the **Z-transform** notation, `z(-1)` is equivalent to `next()` function.
The function `z(...)` method can be used on an `Input` variable to get a time shifted value.

The obective of the minimization is to reduce the error between
`x_{z}_i` that represent one sample of the next position of the mass get from the dataset and 
`x_{z_est}_i` is one sample of the output of our estimator.
The matematical formulation is as follow:
```math
\frac{1}{n} \sum_{i=0}^{n} (x_{z}_i - x_{z_est}_i)^2
```
where `n` represents the number of sample in the dataset.

Finally the function `neuralizeModel` is used to perform the discretization. The parameter of the function is the sampling time and it will be chosen based on the data we have available.

<a name="loaddataset"></a>
### Load the dataset

```python
data_struct = ['time','x','dx','F']
data_folder = './tutorials/datasets/mass-spring-damper/data/'
mass_spring_damper.loadData(name='mass_spring_dataset', source=data_folder, format=data_struct, delimiter=';')
```
Finally, the dataset is loaded. **neu4mes** loads all the files that are in a source folder.

<a name="trainmodel"></a>
### Train the structured neural network
Using the dataset created the training is performed on the model.

```python
mass_spring_damper.trainModel()
```

<a name="testmodel"></a>
### Test the structured neural model
In order to test the results we need to create a input, in this case is defined by:
1. `x` with 5 sample because the sample time is 0.2 and the window of `x`is 1 second.
2. `F` is one sample because only the last sample is needed.

```python
sample = {'F':[0.5], 'x':[0.25, 0.26, 0.27, 0.28, 0.29]}
results = mass_spring_damper(sample)
print(results)
```
The result variable is structured as follow:
```shell
>> {'x_z_est':[0.4]}
```
The value represents the output of our estimator (means the next position of the mass) and is close as possible to `x.next()` get from the dataset.
The network can be tested also using a bigger time window
```python
sample = {'F':[0.5, 0.6], 'x':[0.25, 0.26, 0.27, 0.28, 0.29, 0.30]}
results = mass_spring_damper(sample)
print(results)
```
The value of `x` is build using a moving time window.
The result variable is structured as follow:
```shell
>> {'x_z_est':[0.4, 0.42]}
```
The same output can be generated calling the network using the flag `sampled=True` in this way: 
```python
sample = {'F':[[0.5],[0.6]], 'x':[[0.25, 0.26, 0.27, 0.28, 0.29],[0.26, 0.27, 0.28, 0.29, 0.30]]}
results = mass_spring_damper(sample,sampled=True)
print(results)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="fonlderstructure"></a>
## Structure of the Repository

<a name="neu4mesfolder"></a>
### Neu4mes folder
This folder contains all the neu4mes library files, the main files are the following:
1. __activation.py__ this file contains all the activation functions.
2. __arithmetic.py__ this file contains the aritmetic functions as: +, -, /, *., ^2
3. __fir.py__ this file contains the finite inpulse response filter function. It is a linear operation without bias on the second dimension.
4. __fuzzify.py__ contains the operation for the fuzzification of a variable, commonly used in the local model as activation function.
5. __input.py__ contains the Input class used for create an input for the network.
6. __linear.py__ this file contains the linear function. Typical Linear operation `W*x+b` operated on the third dimension. 
7. __localmodel.py__ this file contains the logic for build a local model.
8. __ouptut.py__ contains the Output class used for create an output for the network.
9. __parameter.py__ contains the logic for create a generic parameters
10. __parametricfunction.py__ are the user custom function. The function can use the pytorch syntax.  
11. __part.py__ are used for selecting part of the data. 
12. __trigonometric.py__ this file contains all the trigonometric functions.
13. __neu4mes.py__ the main file for create the structured network
14. __model.py__ containts the pytorch template model for the structured network

### Tutorial Folder
This folder contains some complex example of the use of the neu4mes fromwork. 
The objective of this folder is demostrate the effectivness of the framework in solving real problems. 
The examples proposed, some of them related of accompanying article, are as follows:
1. Modeling a linear mass spring damper. The obejtive is to estimate the future position and velocity of the mass. 
We consider the system equipped with position sensor and a force sensor.
2. ...

### Tests Folder
This folder contains the unittest of the library in particular each file test a specific functionality.

### Underdev Folder
This folder contains functionality underdevelopment. 
These files presents the new functionalities and the syntax chosen.

### Examples of usage Folder
The files in the examples folder are a collection of the functionality of the library.
Each file present in deep a specific functionality or function of the framework.
This folder is useful to understand the flexibility and capability of the framework.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="shape"></a>
## Overview on signal shape
In this section is explained the shape of the input/output of the network.

<a name="inoutshape"></a>
### Input and output shape from the structured neural model
The structured network can be called in two way:
1. The shape of the inputs not sampled are [total time window size, dim] 
Sampled inputs are reconstructed as soon as the maximum size of the time window is known. 
'dim' represents the size of the input if is not 1 means that the input is a vector.
2. The shape of the sampled inputs are [number of samples = batch, size of time window for a sample, dim]
In the example presented before in the first call the shape for `x` are [1,5,1] for `F` are [1,1,1]
in the second call for `x` are [2,5,1] for `F` are [2,1,1]. In both cases the last dimensions is ignored as the input are scalar.
The output of the structured neural model
The outputs are defined in this way for the different cases:
1. if the shape is [batch, 1, 1] the final two dimensions are collapsed result [batch]
2. if the shape is [batch, window, 1] the last dimension is collapsed result [batch, window]
3. if the shape is [batch, window, dim] the output is equal to [batch, window, dim]
4. if the shape is [batch, 1, dim] the output is equal to [batch, 1, dim]
In the example `x_z_est` has the shape of [1] in the first call and [2] because the the window and the dim were equal to 1.

<a name="elementwiseshape"></a>
### Shape of elementwise Arithmetic, Activation, Trigonometric
The shape and time windows remain unchanged, for the binary operators shape must be equal.
```
input shape = [batch, window, dim] -> output shape = [batch, window, dim]
```

<a name="firshape"></a>
### Shape of Fir input/output
The input must be scalar, the fir compress di time dimension (window) that goes to 1. A vector input is not allowed.
The output dimension of the Fir is moved on the last dimension for create a vector output.
```
input shape = [batch, window, 1] -> output shape = [batch, 1, output dimension of Fir = output_dimension]
```

<a name="linearshape"></a>
### Shape of Linear input/output 
The window remains unchanged and the output dimension is user defined.
```
input shape = [batch, window, dimension] -> output shape = [batch, window, output dimension of Linear = output_dimension]
```

<a name="fuzzyshape"></a>
### Shape of Fuzzy input/output
The function fuzzify the input and creates a vector for output.
The window remains unchanged, input must be scalar. Vector input are not allowed.
```
input shape = [batch, window, 1] -> output shape = [batch, window, number of centers of Fuzzy = len(centers)]
```

<a name="partshape"></a>
### Shape of Part and Select input/output
Part selects a slice of the vector input, the input must be a vector.
Select operation the dimension becomes 1, the input must be a vector.
For both operation if there is a time component it remains unchanged.
```
Part input shape = [batch, window, dimension] -> output shape = [batch, window, selected dimension = [j-i]]
Select input shape = [batch, window, dimension] -> output shape = [batch, window, 1]
```

<a name="timepartshape"></a>
### Shape of TimePart, SimplePart, SampleSelect input/output
The TimePart selects a time window from the signal (works like timewindow `tw([i,j])` but in this the i,j are absolute). 
The SamplePart selects a list of samples from the signal (works like samplewindow `sw([i,j])` but in this the i,j are absolute).
The SampleSelect selects a specific index from the signal (works like zeta operation `z(index)` but in this the index are absolute).
For all the operation the shape remains unchanged.
```
SamplePart input shape = [batch, window, dimension] -> output shape = [batch, selected sample window = [j-i], dimension]
SampleSelect input shape = [batch, window, dimension] -> output shape = [batch, 1, dimension]
TimePart input shape = [batch, window, dimension] -> output shape = [batch, selected time window = [j-i]/sample_time, dimension]
```

<a name="localmodelshape"></a>
### Shape of LocalModel input/output
The local model has two main inputs, activation functions and inputs.
Activation functions have shape of the fuzzy
```
input shape = [batch, window, 1] -> output shape = [batch, window, number of centers of Fuzzy = len(centers)]
```
Inputs go through input function and output function. 
The input shape of the input function can be anything as long as the output shape of the input function have the following dimensions
`[batch, window, 1]` so input functions for example cannot be a Fir with output_dimension different from 1.
The input shape of the output function is `[batch, window, 1]` while the shape of the output of the output functions can be any

<a name="parametersshape"></a>
### Shape of Parameters input/output
Parameter shape are defined as follows `[window = sw or tw/sample_time, dim]` the dimensions can be defined as a tuple and are appended to window
When the time dimension is not defined it is configured to 1

<a name="paramfunshape"></a>
### Shape of Parametric Function input/output
The Parametric functions take inputs and parameters as inputs
Parameter dimensions are the same as defined by the parameters if the dimensions are not defined they will be equal to `[window = 1,dim = 1]`
Dimensions of the inputs inside the parametric function are the same as those managed within the Pytorch framework equal to `[batch, window, dim]`
Output dimensions must follow the same convention `[batch, window, dim]`

<p align="right">(<a href="#readme-top">back to top</a>)</p>