package lightrnn

import (
	"math"
	"testing"

	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

func TestCost(t *testing.T) {
	hidden := &Layer{
		InSize:     2,
		StateSize:  3,
		Activation: Tanh,
		InitState:  anyvec32.MakeVectorData([]float32{0.258427, 0.079246, 0.963164}),
		StateTrans: anyvec32.MakeVectorData([]float32{
			-1.35695, 0.52911, 0.66674,
			-1.55664, 1.98298, -0.25774,
			1.03017, 0.59364, 0.20304,
		}),
		InTrans: anyvec32.MakeVectorData([]float32{
			0.95289, -0.65038,
			0.10296, 0.55944,
			1.12330, -0.16641,
		}),
		Biases: anyvec32.MakeVectorData([]float32{0.58556, 1.11966, 0.74320}),
	}
	out := &OutLayer{
		InSize:     3,
		OutSize:    2,
		Activation: LogSoftmax,
		Weights: anyvec32.MakeVectorData([]float32{
			0.20350, 2.20200, 1.26508,
			1.93698, 2.08915, 1.08922,
		}),
		Biases: anyvec32.MakeVectorData([]float32{-0.54505, 1.88216}),
	}
	r := &RNN{Hidden: []*Layer{hidden}, Output: out}
	samples := sgd.SliceSampleSet{
		seqtoseq.Sample{
			Inputs:  []linalg.Vector{{0.052665, 1.473122}, {-1.846902, 1.185147}},
			Outputs: []linalg.Vector{{1, 0}, {0, 1}},
		},
		seqtoseq.Sample{
			Inputs:  []linalg.Vector{{-0.130720, -0.502619}, {-0.030378, -1.876917}},
			Outputs: []linalg.Vector{{0, 1}, {1, 0}},
		},
	}
	cost := Cost(r, samples).(float32)
	expected := 1.62473212031132
	if math.Abs(float64(cost)-expected) > 1e-5 {
		t.Errorf("expected %f but got %f", expected, cost)
	}
}
