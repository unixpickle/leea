package main

import (
	"math/rand"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
)

const (
	MaxGenLen   = 100
	MinReadable = 0x20
	MaxReadable = 0xfe

	InputScale = 0x10
)

func GenerateSample(b anyrnn.Block) string {
	var res []byte
	c := b.(anynet.Parameterizer).Parameters()[0].Vector.Creator()
	state := b.Start(1)
	lastChar := oneHot(0)
	for i := 0; i < MaxGenLen; i++ {
		numList := c.MakeNumericList(lastChar)
		next := b.Step(state, c.MakeVectorData(numList))
		chr := pick(next.Output())
		state = next.State()
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
	vec = vec.Copy()
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

func oneHot(chr int) []float64 {
	res := make([]float64, 0x100)
	res[chr] = InputScale
	return res
}
