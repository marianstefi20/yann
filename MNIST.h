#pragma once
#include "stdafx.h"
#include <vector>
#include <iostream>
#include "constants.h"


typedef unsigned char uchar;

using namespace std;

uchar** read_mnist_images(string full_path, int& number_of_images, int& image_size);
uchar* read_mnist_labels(string full_path, int& number_of_labels);

vector<vector<double>> mnist_train_images();
vector<vector<double>> mnist_test_images();
vector<int> mnist_train_labels();
vector<int> mnist_test_labels();
vector<vector<int>> mnist_train_labelsV();
vector<vector<int>> mnist_test_labelsV();


/* -------------- Functions added to quickly test a single layer perceptron layer --------------*/
inline vector<int> generateVectorFromLabel(int label, int limit=10) {
	vector<int> vlabel(10, -1);
	vlabel[label] = 1;
	return vlabel;
}

// Loads the train and test data for quick mockup
template <typename T>
void generateTrainData(vector<vector<T>> &trainFeatures, vector<vector<int>> &labels) {
	int nr_of_train_images = 60000;
	int image_size = 784;
	uchar **mnistTrainSet = read_mnist_images(MNIST_TRAIN_IMAGES, nr_of_train_images, image_size);
	uchar *mnistTrainLabelsSet = read_mnist_labels(MNIST_TRAIN_LABELS, nr_of_train_images);
	for (int i = 0; i < nr_of_train_images; i++) {
		trainFeatures.push_back(vector<T>());
		for (int j = 0; j < image_size; j++)
			trainFeatures[i].push_back((T)mnistTrainSet[i][j]);
		labels.push_back(generateVectorFromLabel((int)mnistTrainLabelsSet[i]));
	}
}

template <typename T>
void generateTestData(vector<vector<T>> &testFeatures, vector<vector<int>> &correctLabels) {
	int nr_of_test_images = 10000;
	int image_size = 784;
	uchar **mnistTestSet = read_mnist_images(MNIST_TEST_IMAGES, nr_of_test_images, image_size);
	uchar *mnistTestLabelsSet = read_mnist_labels(MNIST_TEST_LABELS, nr_of_test_images);
	for (int i = 0; i < nr_of_test_images; i++) {
		testFeatures.push_back(vector<T>());
		for (int j = 0; j < image_size; j++)
			testFeatures[i].push_back((T)mnistTestSet[i][j]);
		correctLabels.push_back(generateVectorFromLabel((int)mnistTestLabelsSet[i]));
	}
}
