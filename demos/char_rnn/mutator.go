package main

import (
	"math/rand"
	"sync"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/leea"
)

type Mutator struct {
	Stddev leea.Schedule

	lock  sync.Mutex
	noise []anyvec.Vector
}

func (m *Mutator) Mutate(t int, e leea.Entity, s rand.Source) {
	stddev := m.Stddev.ValueAtTime(t)
	m.lock.Lock()
	defer m.lock.Unlock()
	entity := e.(*Entity)
	if m.noise == nil {
		m.allocNoise(entity)
	}
	p := entity.RNN.Parameters()
	gen := rand.New(s)
	for i, x := range m.noise {
		anyvec.Rand(x, anyvec.Normal, gen)
		x.Scale(entity.RNN.Creator().MakeNumeric(stddev))
		p[i].Add(x)
	}
}

func (m *Mutator) allocNoise(e *Entity) {
	for _, v := range e.RNN.Parameters() {
		m.noise = append(m.noise, v.Copy())
	}
}
