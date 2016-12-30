// Package leea uses the limited-evaluation evolutionary
// algorithm to train parameterized prediction models.
package leea

import (
	"errors"
	"math"
	"math/rand"
)

const (
	DefaultSurvivalRatio = 0.5
)

// A Trainer uses LEEA to train artificial neural nets or
// other parameterized models.
type Trainer struct {
	Evaluator  Evaluator
	Samples    SampleSource
	Population []*Entity

	// MutationSchedule determines the mutation stddev for a
	// given generation.
	MutationSchedule Schedule

	// Inheritance is a number between 0 and 1 that indicates
	// how much of a parent's fitness is passed down to a
	// child.
	Inheritance float64

	// SurvivalRatio is the fraction of individuals who live
	// through a generation to reproduce and/or mutate.
	//
	// If this is 0, DefaultSurvivalRatio is used.
	SurvivalRatio float64

	// Generation is the current generation number.
	// This starts at 0 and is incremented every time Evolve
	// goes through another generation.
	Generation int
}

// MaxFitness returns the maximum fitness across everyone
// in the current generation.
func (t *Trainer) MaxFitness() float64 {
	var m float64
	for _, e := range t.Population {
		m = math.Max(m, e.Fitness)
	}
	return m
}

// Evolve performs evolution and calls f before each
// generation.
// If f returns false, or if the user sends a kill signal,
// Evolve will return.
// It returns an error if the SampleSource fails.
func (t *Trainer) Evolve(f func() bool) error {
	var err error
	loopUntilKilled(func() bool {
		if err != nil {
			return false
		}
		return f()
	}, func() {
		err = t.generation()
	})
	return err
}

func (t *Trainer) generation() error {
	if len(t.Population) == 0 {
		return errors.New("no population")
	}

	samples, err := t.Samples.MiniBatch()
	if err != nil {
		return err
	}
	for _, entity := range t.Population {
		entity.Fitness *= t.Inheritance
		entity.Fitness += t.Evaluator.Evaluate(entity, samples)
	}
	t.reorderEntities()

	n := t.survivorCount()

	// Over-write the dead population with the survivors.
	for i := n; i < len(t.Population); i++ {
		t.Population[i].CrossOver(t.Population[rand.Intn(n)], 0)
	}

	ordering := rand.Perm(len(t.Population))
	for i, j := range ordering[:len(ordering)-1] {
		remainingIdxs := ordering[i+1:]
		otherIdx := remainingIdxs[rand.Intn(len(remainingIdxs))]
		t.Population[j].CrossOver(t.Population[otherIdx], 0.5)
	}

	mutation := t.MutationSchedule.ValueAtTime(t.Generation)
	for _, e := range t.Population {
		e.Mutate(mutation)
	}

	t.Generation++
	return nil
}

func (t *Trainer) reorderEntities() {
	w := newRouletteWheel(t.Population)
	for i := range t.Population {
		t.Population[i] = w.Select()
	}
}

func (t *Trainer) survivorCount() int {
	numSelect := int(t.SurvivalRatio*float64(len(t.Population)) + 0.5)
	if t.SurvivalRatio == 0 {
		numSelect = int(DefaultSurvivalRatio*float64(len(t.Population)) + 0.5)
	}
	if numSelect == 0 {
		return 1
	}
	return numSelect
}

type rouletteWheel struct {
	Entities []*Entity
	Total    float64
}

func newRouletteWheel(entities []*Entity) *rouletteWheel {
	var total float64
	for _, e := range entities {
		total += e.Fitness
	}
	return &rouletteWheel{
		Entities: append([]*Entity{}, entities...),
		Total:    total,
	}
}

func (r *rouletteWheel) Select() *Entity {
	num := rand.Float64() * r.Total
	for i, e := range r.Entities {
		num -= e.Fitness
		if i == len(r.Entities)-1 || num < 0 {
			r.Total -= e.Fitness
			r.Entities[i] = r.Entities[len(r.Entities)-1]
			r.Entities = r.Entities[:len(r.Entities)-1]
			return e
		}
	}
	panic("no entities to select")
}
