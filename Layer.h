#pragma once
#include "Neuron.h"
#include "utils.h"

struct LayerOptions : TrainingOptions {
	vector<vector<int>> * multiLabels;
	int inspect_at;
	double momentum;
};

class Layer {
private: 
	void __setOutputs();  // House keep function that sets a vector of references to the neuron's outputs

protected:
	vector<double> errorToOutputVariation;
	vector<double> outputToNetVariation;
	int width, height;

public:
	Layer(const char * neuronType, int nrOfNeurons, int nrOfInputs);
	~Layer();

	vector<double> outputs;  // populated in constructor
	Layer* previous;
	Layer* next;
	vector<Neuron*> neurons;

	void __initNeurons(const char * neuronType, int nrOfNeurons, int nrOfInputs);
	
	virtual void train(LayerOptions lp);

	//virtual void backPropagate();
	//virtual void forwardPropagate();

	template <typename T>
	vector<vector<T>> test(vector<vector<T>> testData) {
		vector<vector<T>> predictions;
		for (int i = 0; i < this->neurons.size(); i++) {
			predictions.push_back(this->neurons[i]->test(testData));
		}
		return predictions;
	}

	template <typename T1, typename T2>
	void compare(vector<vector<T1>> predictions, vector<vector<T2>> actualTargets) {
		int predictionsNr = predictions.size();
		int vectorNr = predictions[0].size();
		int right = 0;
		for (int i = 0; i < predictionsNr; i++)  
			if (predictions[i] == actualTargets[i])
				right++;
		cout << "Accuracy = " << 1.0 * right / predictionsNr << endl;
	}

	template <typename T1, typename T2>
	double layerLoss(vector<vector<T1>> predictions, vector<vector<T2>> actualTargets) {
		/* Function that computes the loss - in the sense described by the particular loss function found in
		the layer
		:params: predictions - vector containing Layer->neurons vectors that hold all the predictions
		given by that particular neuron in the layer 
		:params: actualTargets - vector containing Layer->neurons vectors that hold all the actualTargets
		*/
		assert(predictions.size() == actualTargets.size());
		int neuronNr = this->neurons.size();
		int vectorNr = predictions.size();
		//struct Loss lossParams;
		double loss = 0.0;

		for (int i = 0; i < vectorNr; i++) {
			for (int j = 0; j < neuronNr; j++) {
				struct Loss lossParams = { predictions[i][j],  (double)actualTargets[i][j] };
				loss += this->neurons[j]->lossFnc(lossParams);		
			}
		}
		return loss;
	}

	template <typename T>
	void forwardPropagate(vector<T> trainInstance) {
		if (this == nullptr) 
			return;
		for (int i = 0; i < this->neurons.size(); i++) {
			this->neurons[i]->_compute(trainInstance);
			if (this->outputs.size() == this->neurons.size()) // if this is the last layer we don't have a bias
				this->outputs[i] = this->neurons[i]->getOutput();
			else
				this->outputs[i + 1] = this->neurons[i]->getOutput();
		}
		this->next->forwardPropagate(this->outputs);
	}

	template <typename T1, typename T2>
	void backPropagate(vector<T1> trainInstance, vector<T2> labels, float learningRate, int nrOfTrainVectors, double momentum=0.0) {
		struct Loss lossParams;

		if (this == nullptr)
			return;

		for (int i = 0; i < (this->neurons).size(); i++)
			this->outputToNetVariation[i] = this->neurons[i]->dActFnc(this->neurons[i]->getSum());
		
		if (this->next == nullptr) 
			for (int i = 0; i < (this->neurons).size(); i++) {
				lossParams.prediction = this->neurons[i]->output;
				lossParams.target = (double)labels[i];
				this->errorToOutputVariation[i] = this->neurons[i]->dLossFnc(lossParams);
			}
		else {
			for (int i = 0; i < (this->neurons).size(); i++) {
				this->errorToOutputVariation[i] = 0.0;
				for (int j = 0; j < (this->next->neurons).size(); j++)
					this->errorToOutputVariation[i] +=
					this->next->errorToOutputVariation[j] *
					this->next->outputToNetVariation[j] *
					this->next->neurons[j]->weights[i + 1];
			}
		}

		
		if (this->previous == nullptr) {
			for (int i = 0; i < (this->neurons).size(); i++)
				for (int k = 0; k < this->neurons[i]->weights.size(); k++) {
					double dW = this->errorToOutputVariation[i] *
						this->outputToNetVariation[i] *
						trainInstance[k];
					double delta = momentum * this->neurons[i]->prevDeltaWeights[k] - learningRate * dW;
					this->neurons[i]->weights[k] += delta;
					this->neurons[i]->prevDeltaWeights[k] = delta;
				}
		}
		else {
			for (int i = 0; i < (this->neurons).size(); i++)
				for (int k = 0; k < this->neurons[i]->weights.size(); k++) {
					double dW = this->errorToOutputVariation[i] *
						this->outputToNetVariation[i] *
						this->previous->outputs[k];
					double delta = momentum * this->neurons[i]->prevDeltaWeights[k] - learningRate * dW;
					this->neurons[i]->weights[k] += delta;
					this->neurons[i]->prevDeltaWeights[k] = delta;
				}
		}

		this->previous->backPropagate(trainInstance, labels, learningRate, nrOfTrainVectors);
	}

	template <typename T>
	void forwardTest(vector<vector<T>> testInstances, vector<vector<T>> &predictions) {
		for (auto testInstance : testInstances) {
			this->forwardPropagate(testInstance);

		}
		if (this == nullptr) {
			predictions = testInstances;
			return;
		}
		vector<vector<T>> tPredictions = vectorTranspose(this->test(testInstances));
		this->next->forwardTest(tPredictions, predictions);
	}

	void saveLayerToFile(const char *filename);
	vector<vector<double>> getWeights();
};

