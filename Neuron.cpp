#include "stdafx.h"

#include <iostream>
#include <vector>
#include <functional>
#include <unordered_set>
#include <map>
#include <algorithm>
#include "assert.h"
#include "utils.h"
#include <time.h>
#include <random>
#include <stdlib.h>
#include <stdio.h>
#include "Neuron.h"


using namespace std;

vector<double> gen_rnd(int nrOfInputs, double uMin, double uMax) {
	random_device rand_dev;
	mt19937 generator(rand_dev());
	uniform_real_distribution<double> distribution(uMin, uMax);
	
	vector<double> weights;
	for (int i = 0; i < nrOfInputs; i++)
		weights.push_back(distribution(generator));
	return weights;
}


Neuron::Neuron(int nrOfInputs, string activationFncLabel, string costFncLabel, double uMin, double uMax) {
	/*
	The main Neuron constructor, inherited by all the different types of neurons
	:param nrOfInputs: The number of inputs, EXCLUDING the bias(<=> the internal nr. of inputs will be +1).
	:param activationFncLabel: The name of the activation function used ("sigmoid", "relu" etc.)
	:param costFncLabel: The name of the cost function used internally by the neuron ("quadratic", "cross" etc.)
	:param uMin: Uniform Distribution lower margin
	:param uMax: Uniform Distribution upper margin
	*/
	srand(time(0));
	this->nrOfInputs = nrOfInputs;
	weights = gen_rnd(nrOfInputs + 1, uMin, uMax);  // + 1 as we keep the bias as weight
	weights[0] = 0.0;
	prevDeltaWeights = vector<double>(nrOfInputs+1, 0.0);
	inputsMask = vector<int>(nrOfInputs + 1, 1.0);
	sum = 0.0;
	activationFnc = ac_fnc[activationFncLabel];
	dActFnc = ac_fnc["derivative_of_" + activationFncLabel];
	lossFnc = loss_fnc[costFncLabel];
	dLossFnc = loss_fnc["derivative_of_" + costFncLabel];
	output = 0.0;
	inputs = nullptr;
}

Neuron::~Neuron()
{
}

vector<double> Neuron::getWeights() {
	return this->weights;
}

vector<int> Neuron::getInputsMask() {
	return this->inputsMask;
}

double Neuron::getSum() {
	return this->sum;
}

double Neuron::getOutput() {
	return this->output;
}

void Neuron::turn(const char *state, vector<int> mask) {
	for (int i = 0; i < mask.size(); i++) 
		this->inputsMask[mask[i]] = (strcmp(state, "on")) ? 1 : (strcmp(state, "off")) ? 0 : 1;
}

void Neuron::resetNeuron() {
	fill(inputsMask.begin(), inputsMask.end(), 1);
	fill(weights.begin(), weights.end(), 0);
	sum = 0.0;
	output = 0.0;
}

