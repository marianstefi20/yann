#include "stdafx.h"
#include "utils.h"
#include <vector>
#include <iostream>
#include <iterator>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <map>

using namespace std;

/* ----------------- Activation functions ----------------- */
// Diferentiable functions
double sigmoid_fn(double x) { return 1.0 / (1.0 + exp(-x)); };
double d_sigmoid_fn(double x) { return exp(-x) / pow(1.0 + exp(-x), 2.0); };

double relu_fn(double x) { return (x < 0.0) ? 0.0 : x; }
double d_relu_fn(double x) { return (x < 0.0) ? 0.0 : 1; };

double e_relu_fn(double x) { return (x <= 0.0) ? 0.1 * (exp(x) - 1) : x; }
double d_e_relu_fn(double x) { return (x <= 0.0) ? 0.1 * exp(x) : 1; }

// Non diferentiable
double hardth_fn(double x) { return (x < 0.0) ? -1.0 : 1.0; };
double heaveside_fn(double x) { return (x < 0.0) ? 0.0 : 1.0; }

// Defining a registry for activation functions
map<string, double (*)(double)> ac_fnc;
void setUpActivation() {
	ac_fnc["sigmoid"] = sigmoid_fn;
	ac_fnc["derivative_of_sigmoid"] = d_sigmoid_fn;
	ac_fnc["relu"] = relu_fn;
	ac_fnc["derivative_of_relu"] = d_relu_fn;
	ac_fnc["erelu"] = e_relu_fn;
	ac_fnc["derivative_of_erelu"] = d_e_relu_fn;
	ac_fnc["sign"] = hardth_fn;
	ac_fnc["heaveside"] = heaveside_fn;
}


/* ----------------- Cost functions ----------------- */
double perceptron_loss(struct Loss l) { 
	double decision = -1.0 * l.prediction * l.target;
	return (decision > 0.0) ? decision : 0;
}

double quadratic_loss(struct Loss l) {
	return 0.5 * (l.prediction - l.target) * (l.prediction - l.target);
}
double derivative_of_quadratic_loss(struct Loss l) {
	return 1.0 * l.prediction - l.target;
}

double cross_entropy_loss(struct Loss l) {
	return -1.0 * (l.target * log(l.prediction) + (1.0 - l.target) * log(1.0 - l.prediction));
}

double derivative_of_cross_entropy_loss(struct Loss l) {
	return -1.0 * (l.target / l.prediction - (1.0 - l.target) / (1.0 - l.prediction));
}

// Defining a registry for the cost functions
map<string, double(*)(struct Loss)> loss_fnc;
void setUpLoss() {
	loss_fnc["perceptron_loss"] = perceptron_loss;
	loss_fnc["quadratic_loss"] = quadratic_loss;
	loss_fnc["derivative_of_quadratic_loss"] = derivative_of_quadratic_loss;
	loss_fnc["cross_entropy_loss"] = cross_entropy_loss;
	loss_fnc["derivative_of_cross_entropy_loss"] = derivative_of_cross_entropy_loss;
}


/* ----------------- Generic display and read fn ----------------- */
void showWeights(vector<double> weights) {
	cout << "The learned hyphothesis is: " << endl;
	cout << "(";
	for (int i = 0; i < weights.size(); i++) {
		cout << weights[i];
		if (i != weights.size() - 1)
			cout << ", ";
	}
	cout << ")" << endl;
}

void showOutputs(vector<double> outputs) {
	for (auto output : outputs)
		cout << output << ", ";
}

void saveModelToFile(const char * filename, vector<double> weights) {
	std::ofstream output_file(filename, std::ios_base::app);
	std::ostream_iterator<double> output_iterator(output_file, ", ");
	std::copy(weights.begin(), weights.end(), output_iterator);
	output_file << endl;
}


void readDataFromFile(
	const char * filename,
	vector<vector<double>> &trainInstances,
	int instanceSize,
	vector <int> &labels)
{
	/**
	Function that reads data from the specified filename and populates two data structures.
	:param filename: The path to the filename to read the data from
	:param trainInstances: The vector where to store the input vector. It will have the format
	[1, x_0, x_1, ...], 1 being to account for the associated bias input (i.e. 1)
	:param instanceSize: The size of the input (for example 2D vectors have 2, 3D have 3 etc.)
	:param labels: Where to store the associated labels for each train vector
	*/
	std::ifstream infile(filename);
	char params[256];
	int lineNr = 0;
	while (infile.getline(params, 256)) {
		if (lineNr == 0) {
			lineNr++;
			continue;
		}
		istringstream ssParams(params);
		double feature;
		int label;
		vector<double> trainInstance;
		trainInstance.push_back(1);  // we are adding the corresponding input for the bias
		for (int i = 0; i < instanceSize; i++) {
			ssParams >> feature;
			ssParams.ignore(2);  //  a comma and a space
			trainInstance.push_back(feature);
		}
		trainInstances.push_back(trainInstance);
		ssParams >> label;
		labels.push_back(label);
	}
}

const char* concat(const char *s1, const char *s2){
	char *result = (char*)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

double exp_decay(double learning_rate, int epoch, int k) {
	return learning_rate * exp(-k * epoch);
}

vector<double> cyclic_learning_rate(string type, int epochs, int cycles, double min, double max, bool decreasing) {
	vector<double> clrs;
	int width = epochs / cycles;	
	double inc = (max - min) / epochs;
	double m;
	for (int i = 0; i < cycles; i++) {
		double clr = 0;
		for (int j = 0; j < width; j++) {
			if (decreasing) 
				max -= inc;
			m = (max - min) * 2.0 / (1.0 * width);
			if (type == "triangle")
				clr = (j <= width / 2) ? (min + j * m) : (max - (j - width / 2)*m);
			else if (type == "cos2")
				clr = min + pow(cos(2 * 3.14159265 / width * i), 2) * (max - min);
			else if (type == "cos")
				clr = min + cos(2 * 3.14159265 / width * i) * (max - min);
			else if (type == "stair") {
				int triangle = (j <= width / 2) ? (min + j * m) : (max - (j - width / 2)*m);
				clr = triangle + (10 - triangle % 10) * (max-min);
			}
			clrs.push_back(clr);
		}
	}
	return clrs;
}