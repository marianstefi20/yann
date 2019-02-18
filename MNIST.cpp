#include "stdafx.h"
#include <iostream>
#include <vector>
#include <fstream>

#include "MNIST.h"
using namespace std;

/* Taken from https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c */

uchar** read_mnist_images(string full_path, int& number_of_images, int& image_size) {
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	typedef unsigned char uchar;

	ifstream file(full_path, ios::binary);

	if (file.is_open()) {
		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

		file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
		file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		image_size = n_rows * n_cols;

		uchar** _dataset = new uchar*[number_of_images];
		for (int i = 0; i < number_of_images; i++) {
			_dataset[i] = new uchar[image_size];
			file.read((char *)_dataset[i], image_size);
		}
		return _dataset;
	}
	else {
		throw runtime_error("Cannot open file `" + full_path + "`!");
	}
}

uchar* read_mnist_labels(string full_path, int& number_of_labels) {
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	typedef unsigned char uchar;

	ifstream file(full_path, ios::binary);

	if (file.is_open()) {
		int magic_number = 0;
		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

		file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

		uchar* _dataset = new uchar[number_of_labels];
		for (int i = 0; i < number_of_labels; i++) {
			file.read((char*)&_dataset[i], 1);
		}
		return _dataset;
	}
	else {
		throw runtime_error("Unable to open file `" + full_path + "`!");
	}
}

/* -- HARD CODED VALUES FOR THE KNOWN DATASETS --  */
int nr_of_train_images = 60000;
int nr_of_test_images = 10000;
int image_size = 784;

vector<vector<double>> mnist_train_images() {
	vector<vector<double>> train_images;
	uchar **mnist_numbers = read_mnist_images(MNIST_TRAIN_IMAGES, nr_of_train_images, image_size);
	for (int i = 0; i < nr_of_train_images; i++) {
		vector<double> image;
		for (int j = 0; j < image_size; j++) {
			image.push_back((int)mnist_numbers[i][j] / 255.0);
		}
		train_images.push_back(image);
	}
	return train_images;
}

vector<int> mnist_train_labels() {
	vector<int> labels;	
	uchar *mnist_labels = read_mnist_labels(MNIST_TRAIN_LABELS, nr_of_test_images);

	for (int i = 0; i < nr_of_test_images; i++)
		labels.push_back((int)mnist_labels);
	return labels;
}

vector<vector<int>> mnist_train_labelsV() {
	vector<vector<int>> vlabels;
	vector<int> num_train_labels = mnist_train_labels();
	for (int i = 0; i < num_train_labels.size(); i++) {
		vector<int> vlabel(10);
		vlabel = { 0 };
		vlabel.at(num_train_labels[i]) = 1;
		vlabels.push_back(vlabel);
	}
	return vlabels;
}

vector<vector<double>> mnist_test_images() {
	vector<vector<double>> test_images;
	uchar **mnist_numbers = read_mnist_images(MNIST_TEST_IMAGES, nr_of_test_images, image_size);
	for (int i = 0; i < nr_of_test_images; i++) {
		vector<double> image;
		for (int j = 0; j < image_size; j++) {
			image.push_back((int)mnist_numbers[i][j] / 255.0);
		}
		test_images.push_back(image);
	}
	return test_images;
}

vector<int> mnist_test_labels() {
	vector<int> labels;
	uchar *mnist_labels = read_mnist_labels(MNIST_TEST_LABELS, nr_of_test_images);

	for (int i = 0; i < nr_of_test_images; i++)
		labels.push_back((int)mnist_labels);
	return labels;
}

vector<vector<int>> mnist_test_labelsV() {
	vector<vector<int>> vlabels;
	vector<int> num_test_labels = mnist_test_labels();
	for (int i = 0; i < num_test_labels.size(); i++) {
		vector<int> vlabel(10);
		vlabel = { 0 };
		vlabel.at(num_test_labels[i]) = 1;
		vlabels.push_back(vlabel);
	}
	return vlabels;
}
