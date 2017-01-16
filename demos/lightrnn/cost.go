package lightrnn

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

// Cost computes cross-entropy loss when the RNN has a
// LogSoftmax output activation.
// The cost is normalized to be an average over all
// timesteps.
// The sample set must be full of seqtoseq.Sample objects,
// all of the same length.
func Cost(r *RNN, s sgd.SampleSet) anyvec.Numeric {
	if s.Len() == 0 {
		return 0
	}
	var inSeqs [][]linalg.Vector
	var outSeqs [][]linalg.Vector
	for i := 0; i < s.Len(); i++ {
		sample := s.GetSample(i).(seqtoseq.Sample)
		inSeqs = append(inSeqs, sample.Inputs)
		outSeqs = append(outSeqs, sample.Outputs)
	}
	steps := len(inSeqs[0])
	n := len(inSeqs)

	state := r.Start(n)

	sum := r.Creator().MakeVector(1)
	for t := 0; t < steps; t++ {
		var inVec, outVec []float64
		for i, s := range inSeqs {
			inVec = append(inVec, s[t]...)
			outVec = append(outVec, outSeqs[i][t]...)
		}
		inData := r.Creator().MakeNumericList(inVec)
		actual := r.Apply(state, r.Creator().MakeVectorData(inData))
		expected := r.Creator().MakeVectorData(r.Creator().MakeNumericList(outVec))
		sum.AddScaler(expected.Dot(actual))
	}

	sum.Scale(r.Creator().MakeNumeric(-1 / float64(n*steps)))
	return anyvec.Sum(sum)
}
