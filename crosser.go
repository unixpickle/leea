package leea

import (
	"math/rand"

	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// A Crosser performs cross-over between learners.
//
// The Cross method takes a keep parameter, which
// indicates the fraction of its own parameters dest
// should retain.
type Crosser interface {
	Cross(dest, source sgd.Learner, keep float64)
}

// A BasicCrosser combines learners by randomly selecting
// individual vector components of the parameter vectors.
type BasicCrosser struct{}

// Cross performs component-wise cross-over.
func (_ BasicCrosser) Cross(dest, source sgd.Learner, keep float64) {
	sourceParams := source.Parameters()
	for i, p := range dest.Parameters() {
		for j, comp := range sourceParams[i].Vector {
			if keep == 0 || rand.Float64() > keep {
				p.Vector[j] = comp
			}
		}
	}
}

func basicCrosserIfNil(c Crosser) Crosser {
	if c == nil {
		return BasicCrosser{}
	}
	return c
}

// A NeuronalCrosser performs cross-over on entire neurons
// at a time in a neuralnet.DenseLayer, using the Default
// Crosser when it does not recognize the type of a
// learner.
//
// Specifically, a NeuronalCrosser knows how to deal with
// a *neuralnet.DenseLayer and a neuralnet.Network.
type NeuronalCrosser struct {
	// Fallback is the Crosser to use if a learner is not a
	// *neuralnet.DenseLayer or neuralnet.Network.
	// If it is nil, BasicCrosser is used.
	Fallback Crosser
}

func (n *NeuronalCrosser) Cross(dest, source sgd.Learner, keep float64) {
	switch dest := dest.(type) {
	case neuralnet.Network:
		source := source.(neuralnet.Network)
		for i, layer := range dest {
			if l, ok := layer.(sgd.Learner); ok {
				sourceLayer := source[i].(sgd.Learner)
				n.Cross(l, sourceLayer, keep)
			}
		}
	case *neuralnet.DenseLayer:
	default:
		n.fallback().Cross(dest, source, keep)
	}
}

func (n *NeuronalCrosser) fallback() Crosser {
	return basicCrosserIfNil(n.Fallback)
}
