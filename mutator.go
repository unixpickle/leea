package leea

import (
	"math/rand"

	"github.com/unixpickle/sgd"
)

// A Mutator applies mutations to sgd.Learners.
// The mutation may depend on the generation, given by t.
type Mutator interface {
	Mutate(t int, l sgd.Learner, s rand.Source)
}

// An AddMutator adds Gaussian noise according to a
// scheduled standard deviation.
type AddMutator struct {
	Stddev Schedule
}

// Mutate adds Gaussian mutations to the parameters.
func (n *AddMutator) Mutate(t int, l sgd.Learner, s rand.Source) {
	r := rand.New(s)
	d := n.Stddev.ValueAtTime(t)
	for _, p := range l.Parameters() {
		for i, comp := range p.Vector {
			p.Vector[i] = comp + r.NormFloat64()*d
		}
	}
}

// A SetMutator randomly assigns a certain fraction of
// the parameters to values sampled from a Gaussian.
type SetMutator struct {
	Fraction Schedule

	// Stddevs must specify the standard deviation for every
	// parameter.
	Stddevs []float64
}

// Mutate replaces some values with randomly-sampled ones.
func (s *SetMutator) Mutate(t int, l sgd.Learner, source rand.Source) {
	r := rand.New(source)
	frac := s.Fraction.ValueAtTime(t)
	for pIdx, p := range l.Parameters() {
		stddev := s.Stddevs[pIdx]
		for i := range p.Vector {
			if r.Float64() < frac {
				p.Vector[i] = r.NormFloat64() * stddev
			}
		}
	}
}
