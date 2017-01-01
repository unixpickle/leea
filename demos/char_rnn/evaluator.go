package main

import (
	"github.com/unixpickle/leea"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn"
)

type Evaluator struct{}

func (_ Evaluator) Evaluate(e *leea.Entity, s sgd.SampleSet) float64 {
	var allInputs [][]linalg.Vector
	var allOutputs [][]linalg.Vector
	for i := 0; i < s.Len(); i++ {
		sample := s.GetSample(i).(Sample)
		allInputs = append(allInputs, sample.InSeq())
		allOutputs = append(allOutputs, sample.OutSeq())
	}
	r := &rnn.Runner{Block: e.Learner.(rnn.Block)}
	var dotSum float64
	var totalLen float64
	for i, seq := range r.RunAll(allInputs) {
		desiredSeq := allOutputs[i]
		for j, vec := range seq {
			desiredVec := desiredSeq[j]
			dotSum += desiredVec.Dot(vec)
			totalLen += 1
		}
	}
	return dotSum / totalLen
}
