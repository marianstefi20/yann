#pragma once
#include <iostream>
#include "Layer.h"

using namespace std;

class Network
{
public:
	int nrOfInputs, nrOfOutputs;
	vector<Layer*> layers;
	vector<vector<double>> features;
	vector<vector<int>> labels;


	Network(int nrOfInputs, int nrOfOutputs);
	~Network();

	void addLayer(const char* neuronType, int nrOfNeurons, int nrOfInputs);
	void train(LayerOptions lo);
	vector<vector<double>> test(vector<vector<double>> testInstances);
};

