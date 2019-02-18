#include "stdafx.h"

#include <iostream>
#include <vector>
#include <string>
#include "Perceptron.h"
#include "MNIST.h"
#include "constants.h"
#include "Layer.h"
#include "Network.h"
#include "utils.h"

using namespace std;

void testMNIST() {
	int nr_of_images = 60000;
	int image_size = 784;
	uchar **mnistTrainSet = read_mnist_images(MNIST_TRAIN_IMAGES, nr_of_images, image_size);
	uchar *mnistTrainLabelsSet = read_mnist_labels(MNIST_TRAIN_LABELS, nr_of_images);
	uchar **mnistTestSet = read_mnist_images(MNIST_TEST_IMAGES, nr_of_images, image_size);
	uchar *mnistTestLabelsSet = read_mnist_labels(MNIST_TEST_LABELS, nr_of_images);

	for (int i = 0; i < 1; i++) {
		cout << "[";
		for (int j = 0; j < 784; j++)
			cout << (int)mnistTrainSet[i][j] << ", ";
		cout << "]" << endl;
		cout << (int)mnistTrainLabelsSet[i] << endl;
	}
}

void test_perceptron_epoch() {
	cout << endl << "Testing a perceptron with automatic epochs number:" << endl;	

	vector<vector<double>> trainInstances;
	vector<int> labels;
	vector<vector<double>> testInstances = { { 1,2,2 },{ 1,7,1 } };
	readDataFromFile(PY_BRIDGE_LINEARLY_SEPARABLE_DATA, trainInstances, 2, labels);

	TrainingOptions tp;
	tp.trainInstances = &trainInstances;
	tp.labels = &labels;
	tp.epochs = 500;
	tp.learningRate = 0.5;
	tp.minimizeLoss = true;
	tp.epsilon = 0.1;
	tp.autoStop = false;
	tp.filename = "";

	Perceptron p = Perceptron(2);
	p.train(tp);
	showWeights(p.getWeights());
	showOutputs(p.test(testInstances));
	cout << endl;

	saveModelToFile(PY_BRIDGE_LINEAR_MODEL, p.getWeights());
}

void test_single_layer_perceptron() {
	vector<vector<double>> trainData;
	vector<vector<int>> testData;
	vector<vector<int>> trainLabels, testLabels;
	generateTrainData(trainData, trainLabels);
	generateTestData(testData, testLabels);
	vector<vector<int>> tTrainLabels = vectorTranspose(trainLabels, 0);

	Layer pL = Layer("perceptron", 10, 784);
	LayerOptions lo;
	lo.trainInstances = &trainData;
	lo.multiLabels = &tTrainLabels;
	lo.minimizeLoss = true;
	lo.epochs = 100;
	lo.learningRate = 1;
	lo.autoStop = false;
	lo.epsilon = 100;
	lo.filename = "";

	pL.train(lo);
	vector<vector<int>> results = pL.test(testData);
	pL.compare(vectorTranspose(results), testLabels);
	pL.saveLayerToFile(PY_BRIDGE_MNIST_LAYER);
}

void test_backprop_linear_separator() {
	vector<vector<double>> trainInstances;
	vector<int> labels;
	vector<vector<double>> testInstances = { { 1,4,-8 },{ 1,5,-7 }, {1,2,5}, {1,2,4}, {1,3,8}, {1, 4, -1} };
	readDataFromFile(PY_BRIDGE_LINEARLY_SEPARABLE_DATA, trainInstances, 2, labels);
	vector<vector<int>> mlabels = multilabel_mapper(labels);

	LayerOptions lo;
	lo.trainInstances = &trainInstances;
	lo.multiLabels = &mlabels;
	lo.minimizeLoss = true;
	lo.epochs = 100000;
	lo.autoStop = false;
	lo.inspect_at = 1000;
	lo.momentum = 0.5;
	LROptions clro{ 0.1, 0.9, 500, "triangle", true };
	lo.learningRateOptions = clro;
	lo.epsilon = 1;
	lo.filename = "";

	Network n = Network(3, 1);
	n.addLayer("sigmoid", 2, 2);
	n.addLayer("sigmoid", 1, 2);
	n.train(lo);

	vector<vector<double>> results;
	for (auto testInstance : testInstances) {
		n.layers[0]->forwardPropagate(testInstance);
		showOutputs(n.layers.back()->outputs);
	}
	cout << endl;
}

void test_xor() {
	vector<vector<double>> trainInstances;
	vector<int> labels;
	vector<vector<double>> testInstances = { { 1,0,0 },{ 1,0,1 },{ 1,1,0 },{ 1,1,1 } };
	readDataFromFile(PY_BRIDGE_XOR_DATA, trainInstances, 2, labels);
	vector<vector<int>> mlabels = multilabel_mapper(labels);

	struct LayerOptions lo;
	lo.trainInstances = &trainInstances;
	lo.multiLabels = &mlabels;
	lo.minimizeLoss = true;
	lo.epochs = 100000;
	LROptions clro{0.1, 0.8, 500, "triangle", true};
	lo.learningRateOptions = clro;
	lo.inspect_at = 1000;
	lo.momentum = 0.9;
	lo.autoStop = false;
	lo.epsilon = 0.1;
	lo.filename = "";

	Network n = Network(3, 1);
	n.addLayer("sigmoid", 2, 2);
	n.addLayer("sigmoid", 1, 2);  
	n.train(lo);

	vector<vector<double>> results;
	for (auto testInstance : testInstances) {
		n.layers[0]->forwardPropagate(testInstance);
		showOutputs(n.layers.back()->outputs);
	}
	cout << endl;
}