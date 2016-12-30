package leea

import (
	"errors"

	"github.com/unixpickle/sgd"
)

// A SampleSource generates mini-batches of samples.
type SampleSource interface {
	MiniBatch() (sgd.SampleSet, error)
}

// A CycleSampleSource produces mini-batches by
// shuffling and cycling through an sgd.SampleSet.
type CycleSampleSource struct {
	// Samples contains the samples to cycle through.
	// This sample set will be shuffled by MiniBatch().
	Samples sgd.SampleSet

	// BatchSize indicates the number of samples to return
	// from MiniBatch().
	BatchSize int

	curIdx int
}

// MiniBatch produces the next batch of samples, shuffling
// the sample set if needed.
func (c *CycleSampleSource) MiniBatch() (sgd.SampleSet, error) {
	if c.BatchSize > c.Samples.Len() {
		return nil, errors.New("batch size exceeds sample count")
	}
	if c.curIdx == 0 || c.curIdx+c.BatchSize > c.Samples.Len() {
		sgd.ShuffleSampleSet(c.Samples)
		c.curIdx = 0
	}
	subset := c.Samples.Subset(c.curIdx, c.curIdx+c.BatchSize)
	c.curIdx += c.BatchSize
	return subset, nil
}
