#pragma once
#include <map>
#include <vector>
#include <string>

using namespace std;

/* ACTIVATION FUNCTIONS */
// Diferentiable functions
double sigmoid_fn(double);
double d_sigmoid_fn(double);

double relu_fn(double);
double d_relu_fn(double);

// Non diferentiable
double hardth_fn(double);
double heaveside_fn(double);

void setUpActivation();
extern map<string, double (*)(double)> ac_fnc;


/* COST FUNCTIONS */
struct Loss { // Holds double prediction and double target. Used to compute loss.
	double prediction;
	double target;
};

double perceptron_loss(struct Loss l);
double quadratic_loss(struct Loss l);
double derivative_of_quadratic_loss(struct Loss l);

void setUpLoss();
extern map <string, double(*)(struct Loss)> loss_fnc;


// Extending operators
// Operator overloading - for making various algorithms cleaner in syntax
template <typename T>
vector<T> operator*(const double & scalar, const vector<T>& values) {
	vector<T> new_values;
	for (auto value : values)
		new_values.push_back(value * scalar);
	return new_values;
};

template <typename T>
vector<T> operator+(const vector<T>& vector1, const vector<T>& vector2) {
	vector<T> new_values;
	for (int i = 0; i < vector1.size(); i++)
		new_values.push_back(vector1[i] + vector2[i]);
	return new_values;
}

template <typename T>
vector<T> operator+=(vector<T> & vector1, const vector<T> & vector2) {
	for (int i = 0; i < vector1.size(); i++)
		vector1[i] += vector2[i];
	return vector1;
}


/* GENERIC FUNCTIONS */
static const char* TRAINING_DATA = "shared/data/train/";
static const char* RESULTS = "shared/data/results/";

void readDataFromFile(
	const char * filename, 
	vector<vector<double>> &trainInstances,
	int instanceSize, 
	vector <int> &labels
);
void showWeights(vector<double> weights);
void showOutputs(vector<double> outputs);
void saveModelToFile(const char * filename, vector<double> weights);
const char* concat(const char *s1, const char *s2);

/* Function that can transpose a vector<vector> array for simpler passing of labels.
Can delete the original vector for saving space.*/
template <typename T>
vector<vector<T>> vectorTranspose(vector<vector<T>> features, int eraseInitial=0) {
	vector<vector<T>> tFeatures(features[0].size());
	for (int i = 0; i < features.size(); i++)
		for (int j = 0; j < features[i].size(); j++)
			tFeatures[j].push_back(features[i][j]);
	if (eraseInitial) {
		for (int i = 0; i < features.size(); i++)
			delete &features[i];
		features.clear();
	}
	return tFeatures;
}

template <typename T>
vector<vector<T>> multilabel_mapper(vector<T> labels) {
	vector<vector<T>> multilabels;
	for (T input : labels)
		multilabels.push_back(vector<T>(1, input));
	return multilabels;
}

double exp_decay(double learningRate, int epoch, int k);
vector<double> cyclic_learning_rate(string type, int epochs, int cycles, double min, double max, bool decreasing=false);