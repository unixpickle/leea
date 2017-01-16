// Package lightrnn implements a lightweight, efficient
// recurrent neural network.
package lightrnn

import (
	"errors"
	"fmt"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/serializer"
)

// State represents the full state of an RNN.
type State []anyvec.Vector

// RNN is a multi-layer recurrent neural network.
type RNN struct {
	Hidden []*Layer
	Output *OutLayer
}

// DeserializeRNN deserializes an RNN.
func DeserializeRNN(d []byte) (*RNN, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) == 0 {
		return nil, errors.New("slice cannot be empty")
	}
	out, ok := slice[0].(*OutLayer)
	if !ok {
		return nil, fmt.Errorf("expected *OutLayer, not %T", slice[0])
	}
	var hidden []*Layer
	for _, x := range slice[1:] {
		h, ok := x.(*Layer)
		if !ok {
			return nil, fmt.Errorf("expected *Layer, not %T", x)
		}
		hidden = append(hidden, h)
	}
	return &RNN{Hidden: hidden, Output: out}, nil
}

// Creator returns the creator to use.
func (r *RNN) Creator() anyvec.Creator {
	return r.Output.Biases.Creator()
}

// Parameters returns all of the vectors that control the
// RNN's behavior.
func (r *RNN) Parameters() []anyvec.Vector {
	res := []anyvec.Vector{r.Output.Weights, r.Output.Biases}
	for _, x := range r.Hidden {
		res = append(res, x.InTrans, x.StateTrans, x.InitState, x.Biases)
	}
	return res
}

// Start returns the initial state for a batch size of n.
func (r *RNN) Start(n int) State {
	var res State
	for _, h := range r.Hidden {
		res = append(res, h.Start(n))
	}
	return res
}

// Apply applies the RNN to a batch, producing a set of
// outputs and updating the state.
func (r *RNN) Apply(s State, in anyvec.Vector) anyvec.Vector {
	nextIn := in
	for i, h := range r.Hidden {
		h.Apply(s[i], nextIn)
		nextIn = s[i]
	}
	return r.Output.Apply(nextIn)
}

// SerializerType returns the unique ID used to serialize
// an RNN with the serializer package.
func (r *RNN) SerializerType() string {
	return "github.com/unixpickle/leea/demos/lightrnn.RNN"
}

// Serialize serializes the RNN.
func (r *RNN) Serialize() ([]byte, error) {
	slice := []serializer.Serializer{r.Output}
	for _, h := range r.Hidden {
		slice = append(slice, h)
	}
	return serializer.SerializeSlice(slice)
}
