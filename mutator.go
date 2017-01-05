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

// An AddMutator adds random noise according to a
// scheduled standard deviation.
type AddMutator struct {
	Stddev  Schedule
	Sampler NumSampler
}

// Mutate adds Gaussian mutations to the parameters.
func (n *AddMutator) Mutate(t int, l sgd.Learner, s rand.Source) {
	r := rand.New(s)
	sampler := n.Sampler.New(r)
	d := n.Stddev.ValueAtTime(t)
	for _, p := range l.Parameters() {
		for i, comp := range p.Vector {
			p.Vector[i] = comp + sampler.Sample()*d
		}
	}
}

// A SetMutator randomly assigns a certain fraction of
// the parameters to values sampled from NumSampler.
type SetMutator struct {
	Fraction Schedule

	// Stddevs must specify the standard deviation for every
	// parameter.
	Stddevs []float64

	// Sampler provides random numbers.
	Sampler NumSampler
}

// Mutate replaces some values with randomly-sampled ones.
func (s *SetMutator) Mutate(t int, l sgd.Learner, source rand.Source) {
	r := rand.New(source)
	sampler := s.Sampler.New(r)
	frac := s.Fraction.ValueAtTime(t)
	for pIdx, p := range l.Parameters() {
		stddev := s.Stddevs[pIdx]
		for i := range p.Vector {
			if r.Float64() < frac {
				p.Vector[i] = sampler.Sample() * stddev
			}
		}
	}
}

// A NumSampler samples random numbers with a variance of
// 1 and a standard deviation of 0.
type NumSampler interface {
	// New creates a new NumSampler with the same behavior as
	// this one, but with a different source.
	New(r *rand.Rand) NumSampler

	// Sample samples a random number.
	Sample() float64
}

// A NormSampler samples from a normal distribution.
//
// The zero value of NormSampler will sample using the
// default rand source.
type NormSampler struct {
	r *rand.Rand
}

// New creates a new NormSampler.
func (n *NormSampler) New(r *rand.Rand) NumSampler {
	return &NormSampler{r: r}
}

// Sample samples from the distribution.
func (n *NormSampler) Sample() float64 {
	if n.r == nil {
		return rand.NormFloat64()
	} else {
		return n.r.NormFloat64()
	}
}
