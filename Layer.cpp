#pragma once
#include "stdafx.h"
#include "Layer.h"
#include "utils.h"
#include "NeuronFactory.h"
#include <fstream>

Layer::Layer(const char * neuronType, int nrOfNeurons, int nrOfInputs) {
	this->previous = nullptr;
	this->next = nullptr;
	this->__initNeurons(neuronType, nrOfNeurons, nrOfInputs);
	this->__setOutputs();
	this->outputToNetVariation = vector<double>(this->neurons.size());
	this->errorToOutputVariation = vector<double>(this->neurons.size());
}

Layer::~Layer() {}

void Layer::__setOutputs() {
	for (auto neuron : this->neurons) 
		this->outputs.push_back(neuron->output);
}

void Layer::__initNeurons(const char *neuronType, int nrOfNeurons, int nrOfInputs) {
	NeuronFactory nf = NeuronFactory();
	while (nrOfNeurons > 0) {
		nrOfNeurons--;
		neurons.push_back(nf.addNeuron(neuronType, nrOfInputs));
	}
}

void Layer::train(LayerOptions lo) {
	for (int i = 0; i<neurons.size(); i++) {
		lo.labels = &(*lo.multiLabels)[i];
		neurons[i]->train(lo);
	}
}

void Layer::saveLayerToFile(const char *filename) {
	ofstream ofs;
	ofs.open(filename, ofstream::out | ofstream::trunc);
	ofs.close();
	for (int i = 0; i<neurons.size(); i++) {
		saveModelToFile(filename, neurons[i]->weights);
	}
	cout << "Saved the model to " << filename <<endl;
}

vector<vector<double>> Layer::getWeights() {
	vector<vector<double>> allWeights;
	for (auto neuron : neurons) 
		allWeights.push_back(neuron->weights);
	return allWeights;
}