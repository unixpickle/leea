// Package leea uses the limited-evaluation evolutionary
// algorithm to train parameterized prediction models.
package leea

import (
	"errors"
	"math"
	"math/rand"
	"runtime"
	"sync"
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
	Selector   Selector
	Mutator    Mutator

	// Crosser is used to perform genetic cross-over.
	// If this is nil, BasicCrosser is used.
	Crosser Crosser

	// DecaySchedule determines how much weight decay should
	// be applied for a given generation.
	// If this is nil, no weight decay is applied.
	DecaySchedule Schedule

	// MutationSchedule determines the mutation stddev for a
	// given generation.
	MutationSchedule Schedule

	// CrossOverSchedule determines the fraction of an
	// individuals parameters that should be updated via
	// cross-over.
	CrossOverSchedule Schedule

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

// BestEntity returns the entity with maximum fitness.
func (t *Trainer) BestEntity() *Entity {
	res := t.Population[0]
	for _, e := range t.Population[1:] {
		if e.Fitness > res.Fitness {
			res = e
		}
	}
	return res
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
		t.Population[i].Set(t.Population[rand.Intn(n)])
	}

	ordering := rand.Perm(len(t.Population))
	crossOver := t.CrossOverSchedule.ValueAtTime(t.Generation)
	for i, j := range ordering[:len(ordering)-1] {
		remainingIdxs := ordering[i+1:]
		otherIdx := remainingIdxs[rand.Intn(len(remainingIdxs))]
		keepRatio := 1 - crossOver
		e := t.Population[j]
		e1 := t.Population[otherIdx]
		e.Fitness = keepRatio*e.Fitness + (1-keepRatio)*e1.Fitness
		t.crosser().Cross(e.Learner, e1.Learner, keepRatio)
	}

	t.mutateAll()
	t.Generation++

	return nil
}

func (t *Trainer) mutateAll() {
	decay := 0.0
	if t.DecaySchedule != nil {
		decay = t.DecaySchedule.ValueAtTime(t.Generation)
	}

	entities := make(chan *Entity, len(t.Population))
	for _, x := range t.Population {
		entities <- x
	}
	close(entities)

	// Mutation benefits from parallelism because normal
	// sampling is expensive.
	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			gen := rand.NewSource(rand.Int63())
			for e := range entities {
				if decay != 0 {
					e.Decay(decay)
				}
				t.Mutator.Mutate(t.Generation, e.Learner, gen)
			}
		}()
	}
	wg.Wait()
}

func (t *Trainer) reorderEntities() {
	t.Selector.SetEntities(t.Population)
	for i := range t.Population {
		t.Population[i] = t.Selector.Select()
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

func (t *Trainer) crosser() Crosser {
	return basicCrosserIfNil(t.Crosser)
}
