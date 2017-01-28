package leea

import (
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
)

// A Crosser performs cross-over between Entity instances.
//
// The Cross method takes a keep parameter, which
// indicates the fraction of its own parameters dest
// should retain.
type Crosser interface {
	Cross(dest, source Entity, keep float64)
}

// A NeuronalCrosser performs cross-over on entire neurons
// at a time in a neural network, using a default crosser
// when it does not recognize the type of a learner.
//
// Specifically, a NeuronalCrosser knows how to deal with
// the following types:
//
//     *anynet.FC
//     anynet.Net
//     anyrnn.Stack
//     *anyrnn.LayerBlock
//     *anyrnn.Vanilla
//     *anyconv.Conv
//
// For the above types, the crosser can unwrap the type
// and apply cross-over to its constituent parts.
type NeuronalCrosser struct{}

// Cross performs cross-over.
// Both entities must be *NetEntity objects.
func (n *NeuronalCrosser) Cross(dest, source Entity, keep float64) {
	n.cross(dest.(*NetEntity).Parameterizer, source.(*NetEntity).Parameterizer, keep)
}

func (n *NeuronalCrosser) cross(dest, source anynet.Parameterizer, keep float64) {
	switch dest := dest.(type) {
	case anynet.Net:
		source := source.(anynet.Net)
		for i, layer := range dest {
			if p, ok := layer.(anynet.Parameterizer); ok {
				sourceLayer := source[i].(anynet.Parameterizer)
				n.cross(p, sourceLayer, keep)
			}
		}
	case *anynet.FC:
		source := source.(*anynet.FC)
		n.crossRows(keep, dest.OutCount, dest.Weights.Vector, source.Weights.Vector,
			dest.Biases.Vector, source.Biases.Vector)
	case *anyconv.Conv:
		source := source.(*anyconv.Conv)
		filterSize := dest.FilterWidth * dest.FilterHeight * dest.InputDepth
		n.crossRows(keep, filterSize, dest.Filters.Vector, source.Filters.Vector,
			dest.Biases.Vector, source.Biases.Vector)
	case anyrnn.Stack:
		source := source.(anyrnn.Stack)
		for i, x := range dest {
			if p, ok := x.(anynet.Parameterizer); ok {
				n.cross(p, source[i].(anynet.Parameterizer), keep)
			}
		}
	case *anyrnn.Vanilla:
		source := source.(*anyrnn.Vanilla)
		n.crossRows(keep, dest.OutCount, dest.InputWeights.Vector,
			source.InputWeights.Vector, dest.StateWeights.Vector,
			source.StateWeights.Vector, dest.Biases.Vector,
			source.Biases.Vector)
	case *anyrnn.LayerBlock:
		if p, ok := dest.Layer.(anynet.Parameterizer); ok {
			pSource := source.(*anyrnn.LayerBlock).Layer.(anynet.Parameterizer)
			n.cross(p, pSource, keep)
		}
	}
}

func (n *NeuronalCrosser) crossRows(keep float64, numRows int, mats ...anyvec.Vector) {
	keepDest := mats[0].Creator().MakeVector(numRows)
	anyvec.Rand(keepDest, anyvec.Uniform, nil)
	anyvec.GreaterThan(keepDest, keepDest.Creator().MakeNumeric(keep))
	takeSrc := keepDest.Copy()
	anyvec.Complement(takeSrc)

	for i := 0; i < len(mats); i += 2 {
		dest := mats[i]
		src := mats[i+1].Copy()
		anyvec.ScaleChunks(dest, keepDest)
		anyvec.ScaleChunks(src, takeSrc)
		dest.Add(src)
	}
}
