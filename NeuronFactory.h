#pragma once

#include "Neuron.h"
#include "Perceptron.h"
#include "Sigmoid.h"
#include "Relu.h"
#include <string.h>

class NeuronFactory {
public :
	NeuronFactory();
	~NeuronFactory();

	Neuron* addNeuron(const char * neuronType, int nrOfInputs);
};