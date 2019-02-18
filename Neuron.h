#pragma once
#include <vector>
#include "Trainable.h"
#include "assert.h"

using namespace std;

struct LROptions {
	double minLr, maxLr;  // the minimum and maximum Learning Rate, if using cyclical learning rate
	int clr_frequency;  // the frequency of the perioadic alternations
	string name;  // special periodic functions - stair, triangle, cos, cos2  (more in utils.cpp)
	bool decreasing;  // should the learningRate decrease in time
};

struct TrainingOptions {
	vector<vector<double>> * trainInstances;
	vector<int> * labels;
	int epochs;
	double learningRate;
	bool minimizeLoss;  // true if we want a good hypothesis
	double epsilon;  // the stop criterion
	bool autoStop;  // Uniform Convergence upper train instances limit
	const char * filename;
	struct LROptions learningRateOptions;
};

struct SPoint {
	double x;
	double y;
};

class Neuron: Trainable {
	friend class Layer;

private:
protected:
	// Properties common to all neurons
	int nrOfInputs;
	double sum;
	double(*activationFnc)(double);
	double(*dActFnc)(double);
	double(*lossFnc)(struct Loss l);
	double(*dLossFnc)(Loss);
	double output;

	/* Specific only to neurons situated in a hidden layer where the inputs are received
	from a previous layer*/
	vector<double> *inputs;  // internal state keeped by a neuron in a hidden layer

	/* Inner function that recomputes the sum and the output when given a particular train instance
	Necessary for computing the Forward Propagation values*/
	template <typename T>
	T _compute(vector<T> trainInstance) {
		this->sum = 0;
		for (int i = 0; i < trainInstance.size(); i++)
			this->sum += (double)trainInstance[i] * this->weights[i] * this->inputsMask[i];
		this->output = this->activationFnc(sum);
		return (T)this->output;
	}
public:
	Neuron(int nrOfInputs, string activationFncLabel, string costFncLabel, double uMin=-2.0/3, double uMax=+2.0/3);
	~Neuron();

	vector<int> inputsMask;
	vector<double> weights;
	vector<double> prevDeltaWeights;
	// Getters for the internal state of the neuron
	vector<double> getWeights();
	double getSum();
	double getOutput();
	vector<int> getInputsMask();

	// Controlling the mask over the weights and the neuron
	void turn(const char * state, vector<int> mask); // either "on" of "off"
	void resetNeuron();

	virtual void train(vector<double> &train_instance, int &label, double& learningRate) = 0;
	virtual void train(vector<vector<double>> &train_instances, vector<int> &labels, double& learningRate) = 0;
	virtual void train(TrainingOptions trainParams) = 0;
	
	template <typename T>
	T test(vector<T> &testInstance) {
		return _compute(testInstance);
	}

	template <typename T>
	vector<T> test(vector<vector<T>> &testInstances) {
		auto testSize = testInstances.size();
		vector<T> results(testSize);

		for(auto i=0;i<testSize;i++)
			results[i] = _compute(testInstances[i]);
		return results;
	}

	template <typename T1, typename T2>
	void test(vector<vector<T1>> &testInstances, vector<T2> &predictions) {
		auto testSize = testInstances.size();
		for (auto i = 0; i < testSize; i++) {
			predictions[i] = _compute(testInstances[i]);
		}
	}
};