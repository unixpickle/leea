package main

import (
	"io/ioutil"
	"strings"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
)

const MaxSentence = 100

func ReadSamples(file string) (sgd.SampleSet, error) {
	contents, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}
	sentences := strings.Split(string(contents), ".")
	var res sgd.SliceSampleSet
	for _, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if len(sentence) >= MaxSentence {
			res = append(res, Sample(sentence[:MaxSentence]))
		}
	}
	return res, nil
}

type Sample string

func (s Sample) OutSeq() []linalg.Vector {
	var res []linalg.Vector
	for _, b := range []byte(s) {
		res = append(res, oneHot(int(b)))
	}
	res = append(res, oneHot(0))
	return res
}

func (s Sample) InSeq() []linalg.Vector {
	var res []linalg.Vector
	res = append(res, oneHot(0))
	for _, b := range []byte(s) {
		res = append(res, oneHot(int(b)))
	}
	return res
}

func oneHot(b int) linalg.Vector {
	res := make(linalg.Vector, 0x100)
	res[b] = 1
	return res
}
