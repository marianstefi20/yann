#pragma once
#include <vector>

using namespace std;

class Trainable {
public:
	virtual void train(vector<double>& train_instance, int& label, double& learningRate) = 0;
	virtual void train(vector<vector<double>>& train_instances, vector<int>& labels, double& learningRate) = 0;
	//virtual double test(vector<double> test_instance) = 0;
	//virtual vector<double> test(vector<vector<double>> test_instances) = 0;
};