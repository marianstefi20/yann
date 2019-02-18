#pragma once

#include "stdafx.h"

#include "Layer.h"
#include <vector>


class InputLayer : public Layer {
private:
	vector<vector<double>> entries;
public:
	InputLayer();
	~InputLayer();
	void setInputs(vector<double> inputs);
	void activateNeurons();
};