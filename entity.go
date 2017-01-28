package leea

import "github.com/unixpickle/anynet"

// An Entity has a set of mutable parameters.
type Entity interface {
	// Decay applies parameter decay to the entity,
	// subtracting rate*x from each parameter x.
	Decay(rate float64)

	// Set copies the contents of e1 into the receiver.
	Set(e1 Entity)
}

// A NetEntity wraps an anynet.Parameterizer and
// implements the entity facilities.
type NetEntity struct {
	anynet.Parameterizer
}

// Decay applies weight decay.
func (n *NetEntity) Decay(r float64) {
	for _, p := range n.Parameterizer.Parameters() {
		p.Vector.Scale(p.Vector.Creator().MakeNumeric(1 - r))
	}
}

// Set copies the parameters from e1.
func (n *NetEntity) Set(e1 Entity) {
	p1 := e1.(*NetEntity).Parameterizer.Parameters()
	for i, x := range n.Parameterizer.Parameters() {
		x.Vector.Set(p1[i].Vector)
	}
}
