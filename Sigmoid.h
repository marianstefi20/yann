#pragma once
#include "stdafx.h"
#include <iostream>
#include <vector>
#include "Neuron.h"
#include "utils.h"

using namespace std;

class Sigmoid : public Neuron {
public:
	Sigmoid(int nrOfInputs);
	~Sigmoid();
	void train(vector<double>& train_instance, int &label, double &learningRate);
	void train(vector<vector<double>>& train_instances, vector<int>& labels, double &learningRate);
	void train(TrainingOptions trainParams);
};
