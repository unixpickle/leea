package main

import (
	"math/rand"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/leea/demos/lightrnn"
)

const (
	MaxGenLen   = 300
	MinReadable = 0x20
	MaxReadable = 0xfe
)

func GenerateSample(r *lightrnn.RNN) string {
	var res []byte
	state := r.Start(1)
	lastChar := oneHot(0)
	for i := 0; i < MaxGenLen; i++ {
		numList := r.Creator().MakeNumericList(lastChar)
		next := r.Apply(state, r.Creator().MakeVectorData(numList))
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

func pick(vec anyvec.Vector) int {
	anyvec.Exp(vec)
	data := vec.Data().([]float32)
	n := rand.Float64()
	for i, x := range data {
		n -= float64(x)
		if n < 0 {
			return i
		}
	}
	return 0xff
}
