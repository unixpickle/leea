package leea

import (
	"fmt"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
)

// An Evaluator evaluates an Entity on a sample batch.
// The result is some measure of fitness, where higher
// fitnesses indicate better performance.
type Evaluator interface {
	Evaluate(e Entity, b anysgd.Batch) float64
}

// NegCost is an Evaluator which computes the negative
// cost for a feed-forward neural net.
//
// It expects batches of the type *anyff.Batch.
type NegCost struct {
	Cost anynet.Cost
}

// Evaluate computes the negative cost.
// In order for this to work, e must be a *NetEntity, the
// net must be an anynet.Layer, and the batch must be an
// *anyff.Batch.
func (n *NegCost) Evaluate(e Entity, s anysgd.Batch) float64 {
	batch := s.(*anyff.Batch)
	net := e.(*NetEntity).Parameterizer.(anynet.Layer)
	trainer := &anyff.Trainer{
		Net:     net,
		Cost:    n.Cost,
		Average: true,
	}
	cost := trainer.TotalCost(batch).Output().Data()
	switch cost := cost.(type) {
	case []float64:
		return -cost[0]
	case []float32:
		return -float64(cost[0])
	default:
		panic(fmt.Sprintf("unsupported numeric type: %T", cost))
	}
}
