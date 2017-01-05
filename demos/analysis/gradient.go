package main

import (
	"fmt"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

func fullGradient(net neuralnet.Network) {
	fmt.Println("Computing full error gradient...")
	g := neuralnet.BatchRGradienter{
		Learner:       net.BatchLearner(),
		CostFunc:      neuralnet.DotCost{},
		MaxBatchSize:  100,
		MaxGoroutines: 1,
	}
	grad := g.Gradient(mnist.LoadTrainingDataSet().SGDSampleSet())
	grad.Scale(1.0 / 60000.0)
	var vals []linalg.Vector
	for _, v := range grad {
		vals = append(vals, v)
	}
	fmt.Println("Gradient second moment:", secondMoment(vals...))
}
