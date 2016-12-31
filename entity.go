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

// Mutate applies normally-distributed noise to the
// parameters with standard deviation d.
func (e *Entity) Mutate(d float64) {
	for _, p := range e.Learner.Parameters() {
		for i, comp := range p.Vector {
			p.Vector[i] = comp + rand.NormFloat64()*d
		}
	}
}

// Set copies the contents of e1 into e.
func (e *Entity) Set(e1 *Entity) {
	e.Fitness = e1.Fitness
	BasicCrosser{}.Cross(e.Learner, e1.Learner, 0)
}
