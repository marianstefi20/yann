#include "stdafx.h"
#include "NeuronFactory.h"

NeuronFactory::NeuronFactory() {
}

NeuronFactory::~NeuronFactory() {
};

Neuron* NeuronFactory::addNeuron(const char *neuronType, int nrOfInputs) {
	Neuron* refToNeuron = nullptr;
	if (strcmp(neuronType, "perceptron") == 0)
		refToNeuron = new Perceptron(nrOfInputs);
	if (strcmp(neuronType, "sigmoid") == 0)
		refToNeuron = new Sigmoid(nrOfInputs);
	if (strcmp(neuronType, "relu") == 0)
		refToNeuron = new Relu(nrOfInputs);
	return refToNeuron;
}