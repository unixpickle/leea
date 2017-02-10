package leea

import (
	"fmt"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
)

// An Evaluator evaluates an Entity on a sample batch.
// The result is some measure of fitness, where higher
// fitnesses indicate better performance.
type Evaluator interface {
	Evaluate(e Entity, b anysgd.Batch) float64
}

// NegCost is an Evaluator which computes the negative
// cost for a feed-forward or recurrent neural network.
//
// It expects batches of the type *anyff.Batch or
// *anys2s.Batch.
type NegCost struct {
	Cost anynet.Cost
}

// Evaluate computes the negative cost.
// In order for this to work, e must be a *NetEntity, the
// net must be an anynet.Layer or an anyrnn.Block, and the
// batch must be an *anyff.Batch or *anys2s.Batch.
func (n *NegCost) Evaluate(e Entity, s anysgd.Batch) float64 {
	var cost anyvec.NumericList

	switch batch := s.(type) {
	case *anyff.Batch:
		net := e.(*NetEntity).Parameterizer.(anynet.Layer)
		trainer := &anyff.Trainer{
			Net:     net,
			Cost:    n.Cost,
			Average: true,
		}
		cost = trainer.TotalCost(batch).Output().Data()
	case *anys2s.Batch:
		block := e.(*NetEntity).Parameterizer.(anyrnn.Block)
		tr := &anys2s.Trainer{
			Func: func(s anyseq.Seq) anyseq.Seq {
				return anyrnn.Map(s, block)
			},
			Cost:    n.Cost,
			Average: true,
		}
		cost = tr.TotalCost(batch).Output().Data()
	default:
		panic(fmt.Sprintf("unsupported batch type: %T", batch))
	}

	switch cost := cost.(type) {
	case []float64:
		return -cost[0]
	case []float32:
		return -float64(cost[0])
	default:
		panic(fmt.Sprintf("unsupported numeric type: %T", cost))
	}
}
