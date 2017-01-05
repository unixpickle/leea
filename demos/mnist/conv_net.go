package main

import "github.com/unixpickle/weakai/neuralnet"

func createConvNet() neuralnet.Network {
	const (
		HiddenSize     = 300
		FilterSize     = 3
		FilterCount    = 5
		FilterStride   = 1
		MaxPoolingSpan = 3
	)

	convOutWidth := (28-FilterSize)/FilterStride + 1
	convOutHeight := (28-FilterSize)/FilterStride + 1

	poolOutWidth := convOutWidth / MaxPoolingSpan
	if convOutWidth%MaxPoolingSpan != 0 {
		poolOutWidth++
	}
	poolOutHeight := convOutWidth / MaxPoolingSpan
	if convOutHeight%MaxPoolingSpan != 0 {
		poolOutHeight++
	}
	net := neuralnet.Network{
		&neuralnet.ConvLayer{
			FilterCount:  FilterCount,
			FilterWidth:  FilterSize,
			FilterHeight: FilterSize,
			Stride:       FilterStride,
			InputWidth:   28,
			InputHeight:  28,
			InputDepth:   1,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.MaxPoolingLayer{
			XSpan:       MaxPoolingSpan,
			YSpan:       MaxPoolingSpan,
			InputWidth:  convOutWidth,
			InputHeight: convOutHeight,
			InputDepth:  FilterCount,
		},
		&neuralnet.DenseLayer{
			InputCount:  poolOutWidth * poolOutHeight * FilterCount,
			OutputCount: HiddenSize,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  HiddenSize,
			OutputCount: 10,
		},
		&neuralnet.SoftmaxLayer{},
	}
	net.Randomize()
	return net
}
