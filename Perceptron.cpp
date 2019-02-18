#include "stdafx.h"
#include <stdlib.h>
#include <vector>
#include <assert.h>
#include <unordered_set>

#include "Perceptron.h"
#include "utils.h"

Perceptron::~Perceptron() {}
Perceptron::Perceptron(int nrOfInputs) :Neuron(nrOfInputs, "sign", "perceptron_loss") {};  // we activate it with the sign fnc

void Perceptron::train(vector<double>& trainInstance, int& label, double& learningRate) {
	if (_compute(trainInstance) * label < 0) {  // _compute modifies the state (sum and output)
		weights += learningRate * label * trainInstance;
	}
}

void Perceptron::train(vector<vector<double>>& trainInstances, vector<int>& labels, double& learningRate) {
	int blockSize = trainInstances.size();
	for (int i = 0; i < blockSize; i++)
		train(trainInstances[i], labels[i], learningRate);
}

void Perceptron::train(TrainingOptions tp) {
	assert((*tp.trainInstances).size() == (*tp.labels).size());
	int epoch = 0;
	double loss;
	Loss lossParams;
	int blockSize = (*tp.trainInstances).size();
	vector<double> predictedLabels(blockSize);

	do {
		if (epoch > tp.epochs) {
			cout << "The perceptron learned for " << tp.epochs << " epochs without reaching a good hyphothesis!" << endl;
			cout << "Total loss= " << loss << endl;
			return;
		}
		loss = 0.0; 
		epoch++;	

		// Actual training of the block (with the actual labels)
		for (int i = 0; i < blockSize; i++)
			train((*tp.trainInstances)[i], (*tp.labels)[i], tp.learningRate);

		// If the minimizeLoss parameter is set, start computing loss (will send the predicted labels)
		if (tp.minimizeLoss) {
			test(*tp.trainInstances, predictedLabels);
			for (int i = 0; i < blockSize; i++) {
				lossParams = {predictedLabels[i], (double)(*tp.labels)[i]};
				loss += lossFnc(lossParams);
			}
		}
	}
	while (loss > tp.epsilon);
	cout << "Finished at epoch " << epoch << ", with loss " << loss <<endl;
}