#include "stdafx.h"

#include "Relu.h"

Relu::~Relu() {}
Relu::Relu(int nrOfInputs) :Neuron(nrOfInputs, "relu", "quadratic_loss") {};  // we activate it with the sign fnc

void Relu::train(vector<double> &train_instance, int &label, double& learningRate) {

}
void Relu::train(vector<vector<double>> &train_instances, vector<int> &labels, double& learningRate) {

}
void Relu::train(TrainingOptions trainParams) {

}