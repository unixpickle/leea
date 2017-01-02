package leea

import (
	"math/rand"

	"github.com/unixpickle/sgd"
)

// An Entity is a single (mutable) individual.
type Entity struct {
	Learner sgd.Learner
	Fitness float64
}

// Decay applies weight decay to the parameters with a
// decay rate r.
func (e *Entity) Decay(r float64) {
	for _, p := range e.Learner.Parameters() {
		for i, x := range p.Vector {
			p.Vector[i] -= x * r
		}
	}
}

// Mutate applies normally-distributed noise to the
// parameters with standard deviation d.
func (e *Entity) Mutate(s rand.Source, d float64) {
	r := rand.New(s)
	for _, p := range e.Learner.Parameters() {
		for i, comp := range p.Vector {
			p.Vector[i] = comp + r.NormFloat64()*d
		}
	}
}

// Set copies the contents of e1 into e.
func (e *Entity) Set(e1 *Entity) {
	e.Fitness = e1.Fitness
	BasicCrosser{}.Cross(e.Learner, e1.Learner, 0)
}
