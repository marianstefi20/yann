#include "stdafx.h"

#include "Sigmoid.h"

Sigmoid::~Sigmoid() {}
Sigmoid::Sigmoid(int nrOfInputs) :Neuron(nrOfInputs, "sigmoid", "cross_entropy_loss") {};  // we activate it with the sign fnc

void Sigmoid::train(vector<double> &train_instance, int &label, double& learningRate) {

}
void Sigmoid::train(vector<vector<double>> &train_instances, vector<int> &labels, double& learningRate) {

}
void Sigmoid::train(TrainingOptions trainParams) {

}