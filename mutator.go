package leea

import (
	"math/rand"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
)

// A Mutator applies mutations to Entity instances.
// The mutation may depend on the generation, given by t.
// Mutators are provided with a random source which they
// may use for random number generation, allowing for
// efficient parallel mutations.
type Mutator interface {
	Mutate(t int, e Entity, r rand.Source)
}

// An AddMutator adds random noise to the parameters of
// entities which implement anynet.Parameterizer.
type AddMutator struct {
	Stddev Schedule
}

// Mutate adds Gaussian mutations to the parameters.
// The e argument must be an sgd.Learner.
func (n *AddMutator) Mutate(t int, e Entity, s rand.Source) {
	parameterizer := e.(anynet.Parameterizer)
	r := rand.New(s)
	d := n.Stddev.ValueAtTime(t)
	for _, p := range parameterizer.Parameters() {
		randVec := p.Vector.Creator().MakeVector(p.Vector.Len())
		anyvec.Rand(randVec, anyvec.Normal, r)
		randVec.Scale(randVec.Creator().MakeNumeric(d))
		p.Vector.Add(randVec)
	}
}

// A SetMutator randomly assigns a certain fraction of
// the parameters of an sgd.Learner to values sampled
// from NumSampler.
type SetMutator struct {
	Fraction Schedule

	// Stddevs must specify the standard deviation for every
	// parameter.
	Stddevs []float64
}

// Mutate replaces some values with randomly-sampled ones.
// The e argument must be an sgd.Learner.
func (s *SetMutator) Mutate(t int, e Entity, source rand.Source) {
	parameterizer := e.(anynet.Parameterizer)
	r := rand.New(source)
	frac := s.Fraction.ValueAtTime(t)
	for pIdx, p := range parameterizer.Parameters() {
		stddev := s.Stddevs[pIdx]

		randVec := p.Vector.Creator().MakeVector(p.Vector.Len())
		anyvec.Rand(randVec, anyvec.Normal, r)
		randVec.Scale(randVec.Creator().MakeNumeric(stddev))
		randVec.Sub(p.Vector)

		mask := p.Vector.Creator().MakeVector(p.Vector.Len())
		anyvec.Rand(mask, anyvec.Uniform, r)
		anyvec.GreaterThan(mask, mask.Creator().MakeNumeric(frac))
		randVec.Mul(mask)

		p.Vector.Add(randVec)
	}
}
