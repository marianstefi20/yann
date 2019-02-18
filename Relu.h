#pragma once
#include "stdafx.h"

#include "Neuron.h"
#include "utils.h"

using namespace std;

class Relu : public Neuron {
public:
	Relu(int nrOfInputs);
	~Relu();

	void train(vector<double>& train_instance, int &label, double &learningRate);
	void train(vector<vector<double>>& train_instances, vector<int>& labels, double &learningRate);
	void train(TrainingOptions trainParams);
};
