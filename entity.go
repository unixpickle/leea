package leea

import "github.com/unixpickle/sgd"

// An Entity is a single (mutable) individual.
type Entity struct {
	Learner sgd.Learner
	Fitness float64
}

// Set copies the contents of e1 into e.
func (e *Entity) Set(e1 *Entity) {
	e.Fitness = e1.Fitness
	BasicCrosser{}.Cross(e.Learner, e1.Learner, 0)
}
