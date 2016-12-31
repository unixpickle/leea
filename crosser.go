package leea

import (
	"math/rand"

	"github.com/unixpickle/sgd"
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
