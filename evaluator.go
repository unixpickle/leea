package leea

import (
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// An Evaluator evaluates an Entity on a set of samples.
// The Evaluate method produces a fitness measure for an
// Entity e on a sample set s.
type Evaluator interface {
	Evaluate(e Entity, s sgd.SampleSet) float64
}

// InvCost is an Evaluator which computes the cost of a
// neural net and returns the reciprocal of said error as
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
// In order for this to work, e must be a *LearnerEntity
// and the learner must be a neuralnet.Network.
func (i *InvCost) Evaluate(e Entity, s sgd.SampleSet) float64 {
	b := e.(*LearnerEntity).Learner.(neuralnet.Network).BatchLearner()
	return 1 / neuralnet.TotalCostBatcher(i.Cost, b, s, i.BatchSize)
}

// NegCost is an Evaluator which computes the negative
// cost of a neural net.
type NegCost struct {
	Cost neuralnet.CostFunc

	// BatchSize is the batch size for evaluating the neural
	// network.
	// If it is 0, the full set of samples is fed in at once.
	BatchSize int
}

// Evaluate computes the negative cost.
// In order for this to work, e must be a *LearnerEntity
// and the learner must be a neuralnet.Network.
//
// The cost is divided by the number of samples.
func (n *NegCost) Evaluate(e Entity, s sgd.SampleSet) float64 {
	b := e.(*LearnerEntity).Learner.(neuralnet.Network).BatchLearner()
	return -neuralnet.TotalCostBatcher(n.Cost, b, s, n.BatchSize) /
		float64(s.Len())
}
