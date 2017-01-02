package leea

import (
	"math/rand"

	"github.com/unixpickle/sgd"
)

// A Mutator applies mutations to sgd.Learners.
type Mutator interface {
	Mutate(l sgd.Learner, amount float64, s rand.Source)
}

// A NormMutator mutates sgd.Learners by adding gaussian
// noise to their parameters.
type NormMutator struct {
	// If Unnormalized is set, then vectors are not rescaled
	// in an attempt to prevent parameter explosion.
	Unnormalized bool

	// MagDrift is the standard deviation of allowed
	// magnitude drift if the magnitude is normalized.
	MagDrift float64
}

// Mutate applies a mutation to l.
func (n *NormMutator) Mutate(l sgd.Learner, stddev float64, s rand.Source) {
	r := rand.New(s)
	for _, p := range l.Parameters() {
		oldMag := p.Vector.Mag()
		for i := range p.Vector {
			p.Vector[i] += r.NormFloat64() * stddev
		}
		if oldMag != 0 && !n.Unnormalized {
			newMag := p.Vector.Mag()
			p.Vector.Scale(oldMag / newMag * (1 + r.NormFloat64()*n.MagDrift))
		}
	}
}
