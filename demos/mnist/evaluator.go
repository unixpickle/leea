package main

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/leea"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

type SoftEvaluator struct{}

func (_ SoftEvaluator) Evaluate(e *leea.Entity, s sgd.SampleSet) float64 {
	b := e.Learner.(neuralnet.Network).BatchLearner()
	c := neuralnet.TotalCostBatcher(neuralnet.DotCost{}, b, s, 0)
	return -c / float64(s.Len())
}

type HardEvaluator struct{}

func (_ HardEvaluator) Evaluate(e *leea.Entity, s sgd.SampleSet) float64 {
	b := e.Learner.(neuralnet.Network).BatchLearner()
	var inputs linalg.Vector
	for i := 0; i < s.Len(); i++ {
		inputs = append(inputs, s.GetSample(i).(neuralnet.VectorSample).Input...)
	}
	out := b.Batch(&autofunc.Variable{Vector: inputs}, s.Len())
	var correct float64
	for i, choices := range autofunc.Split(s.Len(), out) {
		_, max := choices.Output().Max()
		if s.GetSample(i).(neuralnet.VectorSample).Output[max] == 1 {
			correct++
		}
	}
	return correct / float64(s.Len())
}
