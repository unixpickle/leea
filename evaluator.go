package leea

import (
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// An Evaluator evaluates an entity on a set of samples.
// The Evaluate method produces a fitness measure for an
// entity e on a sample set s.
// All fitness values must be non-negative.
type Evaluator interface {
	Evaluate(e *Entity, s sgd.SampleSet) float64
}

// InvCost is an Evaluator which computes the cost of a
// neuralnet and returns the reciprocal of said error as
// the fitness.
type InvCost struct {
	Cost neuralnet.CostFunc

	// BatchSize is the batch size for evaluating the neural
	// network.
	// If it is 0, the full set of samples is fed in at once.
	BatchSize int
}

// Evaluate computes the reciprocal of the cost of the
// entity on the batch.
// In order for this to work, e.Learner must be a
// neuralnet.Network.
func (i *InvCost) Evaluate(e *Entity, s sgd.SampleSet) float64 {
	b := e.Learner.(neuralnet.Network).BatchLearner()
	return 1 / neuralnet.TotalCostBatcher(i.Cost, b, s, i.BatchSize)
}
