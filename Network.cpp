#include "stdafx.h"
#include "Network.h"
#include <random>


Network::Network(int nrOfInputs, int nrOfOutputs) {
	this->nrOfInputs = nrOfInputs;
	this->nrOfOutputs = nrOfOutputs;
}

void Network::addLayer(const char * neuronType, int nrOfNeurons, int nrOfInputs) {
	// add the new layer to the network
	this->layers.push_back(new Layer(neuronType, nrOfNeurons, nrOfInputs));
	// update the links between the layers
	if (this->layers.size() > 1) {
		this->layers.rbegin()[1]->next = this->layers.rbegin()[0];
		this->layers.rbegin()[0]->previous = this->layers.rbegin()[1];
		// the next line adds at the beggining of outputs an additional 1 corresponding to the bias
		this->layers.rbegin()[1]->outputs.insert(this->layers.rbegin()[1]->outputs.begin(), 1);
	}
}

void Network::train(LayerOptions lo) {
	int epoch = 0;
	double loss=20;
	Layer * startLayer = this->layers.front();
	Layer * outputLayer = this->layers.back();
	int trainSize = (*lo.trainInstances).size();

	vector<double> clr = cyclic_learning_rate(
		lo.learningRateOptions.name, 
		lo.epochs, 
		lo.learningRateOptions.clr_frequency, 
		lo.learningRateOptions.minLr, 
		lo.learningRateOptions.maxLr
	);
	//showOutputs(clr);

	random_device rand_dev;
	mt19937 generator(rand_dev());
	uniform_int_distribution<int> distribution(0, trainSize-1);

	do {
		epoch++;
		if (epoch % lo.inspect_at == 0) {
			cout << "Total loss " << loss << "." << endl;
		}
		if (epoch >= lo.epochs) {
			cout << "The network learned for " << lo.epochs << " epochs"
				 << " without reaching to an optimal hypothesis!" << endl;
			cout << "Total loss " << loss << "." << endl;
			return;
		}

		int i = distribution(generator);
		int j = 0;
		while (j < trainSize) {
			startLayer->forwardPropagate((*lo.trainInstances)[i]);  // recursive func. affecting all the layers
			outputLayer->backPropagate(
				(*lo.trainInstances)[i],
				(*lo.multiLabels)[i],
				clr[epoch],
				(*lo.trainInstances).size(),
				lo.momentum);  // recursive func. affecting all the layers
			j++;
		}
		vector<vector<double>> predictions;
		for (int i = 0; i < trainSize; i++) {
			startLayer->forwardPropagate((*lo.trainInstances)[i]);
			predictions.push_back(outputLayer->outputs);
		}
		loss = outputLayer->layerLoss(predictions, *lo.multiLabels);
	} while (loss > lo.epsilon);

	if (loss < lo.epsilon) {
		cout << endl << "Finished at epoch " << epoch << "." << endl;
		for (int i = 0; i < this->layers.size(); i++) {
			for (int j = 0; j < this->layers[i]->neurons.size(); j++) {
				cout << "Neuron " << j << ", strat " << i << ": ";
				for (auto weight : this->layers[i]->neurons[j]->weights) 
					cout << weight << ", ";
				cout << endl;
			}
		}
	}
}

vector<vector<double>> Network::test(vector<vector<double>> testInstances) {
	vector<vector<double>> results;
	Layer * startLayer = this->layers.front();
	Layer * outputLayer = this->layers.back();
	for (auto test : testInstances) {
		startLayer->forwardPropagate(test);
		results.push_back(outputLayer->outputs);
	}
	return results;
}


Network::~Network()
{
}
