// NeuralNetwork.cpp : Defines the entry point for the console application.
//

#include <functional>

#include "stdafx.h"
#include "Neuron.h"
#include "MNIST.h"
#include "utils.h"

#include "tests.h"

int main()
{
	setUpActivation();
	setUpLoss();

	int iPrev = _CrtSetReportMode(_CRT_ASSERT, 0);

	//testMNIST();
	//test_perceptron_epoch();
	//test_single_layer_perceptron();
	test_backprop_linear_separator();
	test_xor();

	_CrtSetReportMode(_CRT_ASSERT, iPrev);

	return 0;
}

