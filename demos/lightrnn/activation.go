package lightrnn

import (
	"fmt"

	"github.com/unixpickle/anyvec"
)

// An Activation is a neural activation function.
type Activation int

const (
	Tanh Activation = iota
	LogSoftmax
)

// Apply applies the activation function to a batch of
// inputs, where each input is vecSize components.
func (a Activation) Apply(in anyvec.Vector, vecSize int) {
	if in.Len()%vecSize != 0 {
		panic("vecSize does not divide input length")
	}
	switch a {
	case Tanh:
		anyvec.Tanh(in)
	case LogSoftmax:
		anyvec.LogSoftmax(in, vecSize)
	default:
		panic(fmt.Sprintf("unknown activation: %d", a))
	}
}
