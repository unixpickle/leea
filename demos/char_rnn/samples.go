package main

import (
	"io/ioutil"
	"strings"

	"github.com/unixpickle/anynet/anysgd"
	charrnn "github.com/unixpickle/char-rnn"
	"github.com/unixpickle/essentials"
)

const MaxSentence = 100

func ReadSamples(file string) (anysgd.SampleList, error) {
	contents, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, essentials.AddCtx("read samples", err)
	}
	sentences := strings.Split(string(contents), ".")
	var res charrnn.SampleList
	for _, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if len(sentence) >= MaxSentence {
			res = append(res, []byte(sentence[:MaxSentence]))
		}
	}
	return res, nil
}
