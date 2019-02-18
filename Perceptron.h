#pragma once
#include "stdafx.h"
#include <iostream>
#include <vector>
#include "Neuron.h"
#include "utils.h"

using namespace std;

class Perceptron : public Neuron {
public:
	Perceptron(int nrOfInputs);
	~Perceptron();
	void train(vector<double>& trainInstance, int& output, double& learningRate);
	void train(vector<vector<double>>& trainInstances, vector<int>& labels, double& learningRate);
	void train(TrainingOptions trainParams);

};