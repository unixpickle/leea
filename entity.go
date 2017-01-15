package leea

import "github.com/unixpickle/sgd"

// An Entity has a set of mutable parameters.
type Entity interface {
	// Decay applies parameter decay to the entity,
	// subtracting rate*x from each parameter x.
	Decay(rate float64)

	// Set copies the contents of e1 into the receiver.
	Set(e1 Entity)
}

// A LearnerEntity wraps an sgd.Learner and implements the
// entity facilities.
type LearnerEntity struct {
	sgd.Learner
}

// Decay applies weight decay.
func (l *LearnerEntity) Decay(r float64) {
	for _, p := range l.Learner.Parameters() {
		for i, x := range p.Vector {
			p.Vector[i] -= x * r
		}
	}
}

// Set copies the parameters from e1.
func (l *LearnerEntity) Set(e1 Entity) {
	p1 := e1.(*LearnerEntity).Learner.Parameters()
	for i, x := range l.Learner.Parameters() {
		copy(x.Vector, p1[i].Vector)
	}
}
