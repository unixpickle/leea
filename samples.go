package leea

import (
	"errors"

	"github.com/unixpickle/anynet/anysgd"
)

// A SampleSource produces mini-batches of samples.
type SampleSource interface {
	MiniBatch() (anysgd.SampleList, error)
}

// A CycleSampleSource produces mini-batches by
// shuffling and cycling through an anysgd.SampleList.
type CycleSampleSource struct {
	// Samples contains the samples to cycle through.
	// It will be shuffled by MiniBatch().
	Samples anysgd.SampleList

	// BatchSize indicates the number of samples to return
	// from MiniBatch().
	BatchSize int

	curIdx int
}

// MiniBatch produces the next batch of samples, shuffling
// the sample set if needed.
func (c *CycleSampleSource) MiniBatch() (anysgd.SampleList, error) {
	if c.BatchSize > c.Samples.Len() {
		return nil, errors.New("batch size exceeds sample count")
	}
	if c.curIdx == 0 || c.curIdx+c.BatchSize > c.Samples.Len() {
		anysgd.Shuffle(c.Samples)
		c.curIdx = 0
	}
	subset := c.Samples.Slice(c.curIdx, c.curIdx+c.BatchSize)
	c.curIdx += c.BatchSize
	return subset, nil
}
