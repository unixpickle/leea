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

// CrossOver updates the parameters in e by borrowing some
// parameters from e1.
// The keepRatio determines how many of e's parameters are
// retained rather than being taken from e1.
// For instance, if keepRatio is 0, then e is set entirely
// to e1.
//
// The fitness of e is updated to reflect the fitnesses of
// both e and e1.
func (e *Entity) CrossOver(e1 *Entity, keepRatio float64) {
	e.Fitness = e.Fitness*keepRatio + e1.Fitness*(1-keepRatio)
	e1p := e1.Learner.Parameters()
	for i, p := range e.Learner.Parameters() {
		p1 := e1p[i]
		for j, comp := range p1.Vector {
			if rand.Float64() > keepRatio {
				p.Vector[j] = comp
			}
		}
	}
}
