package main

import (
	"math"
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

const (
	MaxGenLen   = 300
	MinReadable = 0x20
	MaxReadable = 0xfe
)

func GenerateSample(b rnn.StackedBlock) string {
	var res []byte
	r := &rnn.Runner{Block: b}
	lastChar := oneHot(0)
	for i := 0; i < MaxGenLen; i++ {
		next := r.StepTime(lastChar)
		chr := pick(next)
		if chr == 0 {
			break
		}
		if chr < MinReadable || chr > MaxReadable {
			chr = '?'
		}
		lastChar = oneHot(chr)
		res = append(res, byte(chr))
	}
	return string(res)
}

func pick(vec linalg.Vector) int {
	n := rand.Float64()
	for i, x := range vec {
		n -= math.Exp(x)
		if n < 0 {
			return i
		}
	}
	return 0xff
}
