package lightrnn

import (
	"math"
	"math/rand"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/serializer"
)

func init() {
	var l Layer
	serializer.RegisterTypedDeserializer(l.SerializerType(), DeserializeLayer)
	var o OutLayer
	serializer.RegisterTypedDeserializer(o.SerializerType(), DeserializeOutLayer)
}

// Layer is a recurrent layer of a multi-layer RNN.
type Layer struct {
	InSize     int
	StateSize  int
	Activation Activation

	// InitState contains the initial state.
	InitState anyvec.Vector

	// StateTrans contains the matrix that is applied
	// to the previous state.
	StateTrans anyvec.Vector

	// InTrans contains the matrix that is applied to
	// the input.
	InTrans anyvec.Vector

	// Biases is the bias vector which is added to the
	// resulting state before it is squashed.
	Biases anyvec.Vector
}

// NewLayer creates a Layer with randomized weights.
func NewLayer(c anyvec.Creator, in, state int, act Activation) *Layer {
	return &Layer{
		InSize:     in,
		StateSize:  state,
		Activation: act,
		InitState:  c.MakeVector(state),
		StateTrans: randomMatrix(c, state, state),
		InTrans:    randomMatrix(c, in, state),
		Biases:     c.MakeVector(state),
	}
}

// DeserializeLayer deserializes a Layer.
func DeserializeLayer(d []byte) (*Layer, error) {
	var inSize, stateSize, activation serializer.Int
	var initState, stateTrans, inTrans, biases *anyvecsave.S
	err := serializer.DeserializeAny(d, &inSize, &stateSize, &activation, &initState,
		&stateTrans, &inTrans, &biases)
	if err != nil {
		return nil, err
	}
	return &Layer{
		InSize:     int(inSize),
		StateSize:  int(stateSize),
		Activation: Activation(activation),
		InitState:  initState.Vector,
		StateTrans: stateTrans.Vector,
		InTrans:    inTrans.Vector,
		Biases:     biases.Vector,
	}, nil
}

// Start repeats the initial state the given number of
// times.
func (l *Layer) Start(n int) anyvec.Vector {
	return repeatVec(l.InitState, n)
}

// Apply applies the layer to a batch of inputs.
// The states vector is overwritten to the new value.
func (l *Layer) Apply(states, inputs anyvec.Vector) {
	n := states.Len() / l.StateSize
	one := states.Creator().MakeNumeric(1)
	zero := states.Creator().MakeNumeric(0)

	trans := &anyvec.Matrix{Data: l.StateTrans, Rows: l.StateSize, Cols: l.StateSize}
	data := &anyvec.Matrix{Data: states.Copy(), Rows: n, Cols: l.StateSize}
	out := &anyvec.Matrix{Data: states, Rows: n, Cols: l.StateSize}

	out.Product(false, true, one, data, trans, zero)

	trans = &anyvec.Matrix{Data: l.InTrans, Rows: l.StateSize, Cols: l.InSize}
	data = &anyvec.Matrix{Data: inputs, Rows: n, Cols: l.InSize}
	out.Product(false, true, one, data, trans, one)

	states.Add(repeatVec(l.Biases, n))
	l.Activation.Apply(states, l.StateSize)
}

// SerializerType returns the unique ID used to serialize
// a Layer with the serializer package.
func (l *Layer) SerializerType() string {
	return "github.com/unixpickle/leea/demos/lightrnn.Layer"
}

// Serialize serializes the Layer.
func (l *Layer) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		serializer.Int(l.InSize),
		serializer.Int(l.StateSize),
		serializer.Int(l.Activation),
		&anyvecsave.S{Vector: l.InitState},
		&anyvecsave.S{Vector: l.StateTrans},
		&anyvecsave.S{Vector: l.InTrans},
		&anyvecsave.S{Vector: l.Biases},
	)
}

// OutLayer is the output layer of a multi-layer RNN.
type OutLayer struct {
	InSize     int
	OutSize    int
	Activation Activation

	Weights anyvec.Vector
	Biases  anyvec.Vector
}

// NewOutLayer creates a randomized OutLayer.
func NewOutLayer(c anyvec.Creator, in, out int, act Activation) *OutLayer {
	return &OutLayer{
		InSize:     in,
		OutSize:    out,
		Activation: act,
		Weights:    randomMatrix(c, in, out),
		Biases:     c.MakeVector(out),
	}
}

// DeserializeOutLayer deserializes an OutLayer.
func DeserializeOutLayer(d []byte) (*OutLayer, error) {
	var inSize, outSize, activation serializer.Int
	var weights, biases *anyvecsave.S
	err := serializer.DeserializeAny(d, &inSize, &outSize, &activation, &weights, &biases)
	if err != nil {
		return nil, err
	}
	return &OutLayer{
		InSize:     int(inSize),
		OutSize:    int(outSize),
		Activation: Activation(activation),
		Weights:    weights.Vector,
		Biases:     biases.Vector,
	}, nil
}

// Apply applies the layer to some inputs, producing an
// output vector.
func (o *OutLayer) Apply(in anyvec.Vector) anyvec.Vector {
	n := in.Len() / o.InSize
	outVec := in.Creator().MakeVector(o.OutSize * n)
	one := in.Creator().MakeNumeric(1)
	zero := in.Creator().MakeNumeric(0)

	trans := &anyvec.Matrix{Data: o.Weights, Rows: o.OutSize, Cols: o.InSize}
	data := &anyvec.Matrix{Data: in, Rows: n, Cols: o.InSize}
	out := &anyvec.Matrix{Data: outVec, Rows: n, Cols: o.OutSize}

	out.Product(false, true, one, data, trans, zero)
	outVec.Add(repeatVec(o.Biases, n))
	o.Activation.Apply(outVec, o.OutSize)

	return outVec
}

// SerializerType returns the unique ID used to serialize
// an OutLayer with the serializer package.
func (o *OutLayer) SerializerType() string {
	return "github.com/unixpickle/leea/demos/lightrnn.OutLayer"
}

// Serialize serializes an OutLayer.
func (o *OutLayer) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		serializer.Int(o.InSize),
		serializer.Int(o.OutSize),
		serializer.Int(o.Activation),
		&anyvecsave.S{Vector: o.Weights},
		&anyvecsave.S{Vector: o.Biases},
	)
}

func randomMatrix(c anyvec.Creator, inCount, outCount int) anyvec.Vector {
	stddev := 1 / math.Sqrt(float64(inCount))
	data := make([]float64, inCount*outCount)
	for i := range data {
		data[i] = rand.NormFloat64() * stddev
	}
	return c.MakeVectorData(c.MakeNumericList(data))
}

func repeatVec(v anyvec.Vector, n int) anyvec.Vector {
	var a []anyvec.Vector
	for i := 0; i < n; i++ {
		a = append(a, v)
	}
	return v.Creator().Concat(a...)
}
