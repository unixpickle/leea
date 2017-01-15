package leea

import (
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weightnorm"
)

// A Crosser performs cross-over between Entity instances.
//
// The Cross method takes a keep parameter, which
// indicates the fraction of its own parameters dest
// should retain.
type Crosser interface {
	Cross(dest, source Entity, keep float64)
}

// A LearnerCrosser combines entities which implement
// sgd.Learner by randomly selecting individual vector
// components of the parameter vectors.
type LearnerCrosser struct{}

// Cross performs component-wise cross-over.
func (l LearnerCrosser) Cross(dest, source Entity, keep float64) {
	sourceParams := source.(sgd.Learner).Parameters()
	for i, p := range dest.(sgd.Learner).Parameters() {
		for j, comp := range sourceParams[i].Vector {
			if keep == 0 || rand.Float64() > keep {
				p.Vector[j] = comp
			}
		}
	}
}

// A NeuronalCrosser performs cross-over on entire neurons
// at a time in a neural network, using a default crosser
// when it does not recognize the type of a learner.
//
// Specifically, a NeuronalCrosser knows how to deal with
// the following types:
//
//     *neuralnet.DenseLayer
//     neuralnet.Network
//     rnn.StackedBlock
//     *rnn.StateOutBlock
//     *rnn.NetworkBlock
//     *neuralnet.ConvLayer
//     *weightnorm.Norm
//
// For the above types, the crosser can unwrap the type
// and apply cross-over to its constituent parts.
type NeuronalCrosser struct {
	// Fallback is the Crosser to use if a learner is not a
	// *neuralnet.DenseLayer or neuralnet.Network.
	// If it is nil, BasicCrosser is used.
	Fallback Crosser
}

// Cross performs cross-over, taking neural structures
// into account whenever possible.
// Both entities must be *LearnerEntity instances.
func (n *NeuronalCrosser) Cross(dest, source Entity, keep float64) {
	n.crossLearners(dest.(*LearnerEntity).Learner, source.(*LearnerEntity).Learner, keep)
}

func (n *NeuronalCrosser) crossLearners(dest, source sgd.Learner, keep float64) {
	switch dest := dest.(type) {
	case neuralnet.Network:
		source := source.(neuralnet.Network)
		for i, layer := range dest {
			if l, ok := layer.(sgd.Learner); ok {
				sourceLayer := source[i].(sgd.Learner)
				n.crossLearners(l, sourceLayer, keep)
			}
		}
	case *neuralnet.DenseLayer:
		source := source.(*neuralnet.DenseLayer)
		neuronCount := dest.Weights.Rows
		inputCount := dest.Weights.Cols
		for i := 0; i < neuronCount; i++ {
			if rand.Float64() < keep {
				startIdx := i * inputCount
				endIdx := (i + 1) * inputCount
				copy(dest.Weights.Data.Vector[startIdx:endIdx],
					source.Weights.Data.Vector[startIdx:endIdx])
				dest.Biases.Var.Vector[i] = source.Biases.Var.Vector[i]
			}
		}
	case *neuralnet.ConvLayer:
		source := source.(*neuralnet.ConvLayer)
		for i, x := range dest.Filters {
			if rand.Float64() < keep {
				copy(x.Data, source.Filters[i].Data)
				dest.Biases.Vector[i] = source.Biases.Vector[i]
			}
		}
	case rnn.StackedBlock:
		source := source.(rnn.StackedBlock)
		for i, x := range dest {
			if l, ok := x.(sgd.Learner); ok {
				n.crossLearners(l, source[i].(sgd.Learner), keep)
			}
		}
	case *rnn.StateOutBlock:
		source := source.(*rnn.StateOutBlock).Block
		if l, ok := dest.Block.(sgd.Learner); ok {
			n.crossLearners(l, source.(sgd.Learner), keep)
		}
	case *rnn.NetworkBlock:
		source := source.(*rnn.NetworkBlock)
		l1 := &variableLearner{Variable: dest.Parameters()[0]}
		l2 := &variableLearner{Variable: source.Parameters()[0]}
		n.crossLearners(l1, l2, keep)
		n.crossLearners(dest.Network(), source.Network(), keep)
	case *weightnorm.Norm:
		source := source.(*weightnorm.Norm)
		for i, destWeights := range dest.Weights {
			destMags := dest.Mags[i]
			numRows := len(destMags.Vector)
			numCols := len(destWeights.Vector) / len(destMags.Vector)
			sourceWeights := source.Weights[i]
			sourceMags := source.Mags[i]
			for j := 0; j < numRows; j++ {
				if rand.Float64() < keep {
					s, e := j*numCols, (j+1)*numCols
					copy(destWeights.Vector[s:e], sourceWeights.Vector[s:e])
					destMags.Vector[j] = sourceMags.Vector[j]
				}
			}
		}
		skip := len(dest.Weights) + len(dest.Mags)
		sp := source.Parameters()[skip:]
		for i, x := range dest.Parameters()[skip:] {
			l1 := &variableLearner{Variable: x}
			l2 := &variableLearner{Variable: sp[i]}
			n.crossLearners(l1, l2, keep)
		}
	default:
		n.fallback().Cross(&LearnerEntity{Learner: dest},
			&LearnerEntity{Learner: source}, keep)
	}
}

func (n *NeuronalCrosser) fallback() Crosser {
	if n.Fallback == nil {
		return LearnerCrosser{}
	}
	return n.Fallback
}

type variableLearner struct {
	Variable *autofunc.Variable
}

func (i *variableLearner) Parameters() []*autofunc.Variable {
	return []*autofunc.Variable{i.Variable}
}
