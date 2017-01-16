package main

import (
	"github.com/unixpickle/leea"
	"github.com/unixpickle/leea/demos/lightrnn"
)

type Entity struct {
	RNN *lightrnn.RNN
}

func (e *Entity) Decay(rate float64) {
	for _, p := range e.RNN.Parameters() {
		p.Scale(e.RNN.Creator().MakeNumeric(1 - rate))
	}
}

func (e *Entity) Set(e1 leea.Entity) {
	p1 := e1.(*Entity).RNN.Parameters()
	for i, x := range e.RNN.Parameters() {
		x.Set(p1[i])
	}
}
