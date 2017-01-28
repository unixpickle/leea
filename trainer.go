// Package leea uses the limited-evaluation evolutionary
// algorithm to train parameterized prediction models.
package leea

import (
	"errors"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"

	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/rip"
)

const (
	DefaultSurvivalRatio = 0.5
)

// A Trainer uses LEEA to train artificial neural nets or
// other parameterized models.
type Trainer struct {
	Evaluator  Evaluator
	Samples    SampleSource
	Fetcher    anysgd.Fetcher
	Population []*FitEntity
	Selector   Selector
	Mutator    Mutator
	Crosser    Crosser

	// DecaySchedule determines how much weight decay should
	// be applied for a given generation.
	// If this is nil, no weight decay is applied.
	DecaySchedule Schedule

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

	// Elitism specifies the number of individuals who are
	// untouched by mutation and cross-over.
	Elitism int

	// Generation is the current generation number.
	// This starts at 0 and is incremented every time Evolve
	// goes through another generation.
	Generation int
}

// FitnessScale is the number by which fitnesses should be
// divided to get the "running average" fitness.
// Basically, it accounts for the geometric series with
// decay rate given by t.Inheritance.
func (t *Trainer) FitnessScale() float64 {
	if t.Generation < 2 {
		return 1
	}
	if t.Inheritance == 1 {
		return float64(t.Generation - 1)
	}

	// Use geometric series if it's accurate enough.
	epsilon := math.Nextafter(1, 2) - 1
	if math.Pow(t.Inheritance, float64(t.Generation-1)) < epsilon {
		return 1 / (1 - t.Inheritance)
	}

	sum := 1.0
	for i := 1; i < t.Generation; i++ {
		sum *= t.Inheritance
		sum += 1
	}
	return sum
}

// MaxFitness returns the maximum fitness across everyone
// in the current generation.
func (t *Trainer) MaxFitness() float64 {
	m := math.Inf(-1)
	for _, e := range t.Population {
		m = math.Max(m, e.Fitness)
	}
	return m
}

// MeanFitness returns the mean fitness.
func (t *Trainer) MeanFitness() float64 {
	var sum float64
	for _, e := range t.Population {
		sum += e.Fitness
	}
	return sum / float64(len(t.Population))
}

// BestEntity returns the entity with maximum fitness.
func (t *Trainer) BestEntity() *FitEntity {
	res := t.Population[0]
	for _, e := range t.Population[1:] {
		if e.Fitness > res.Fitness {
			res = e
		}
	}
	return res
}

// Evolve performs evolution.
// Before every generation, f is called.
// Evolution stops when f returns false or when the user
// sends an interrupt signal.
// This returns an error if fetching samples fails.
func (t *Trainer) Evolve(f func() bool) error {
	killSig := rip.NewRIP()

	for !killSig.Done() {
		if !f() {
			return nil
		}
		if killSig.Done() {
			return nil
		}
		if err := t.generation(); err != nil {
			return err
		}
	}

	return nil
}

func (t *Trainer) generation() error {
	if len(t.Population) == 0 {
		return errors.New("no population")
	}

	samples, err := t.Samples.MiniBatch()
	if err != nil {
		return err
	}
	batch, err := t.Fetcher.Fetch(samples)
	if err != nil {
		return err
	}

	for _, entity := range t.Population {
		entity.Fitness *= t.Inheritance
		entity.Fitness += t.Evaluator.Evaluate(entity.Entity, batch)
	}
	t.reorderEntities()

	n := t.survivorCount()

	// Overwrite the dead population with the survivors.
	for i := n; i < len(t.Population); i++ {
		source := t.Population[rand.Intn(n)]
		dest := t.Population[i]
		dest.Entity.Set(source.Entity)
		dest.Fitness = source.Fitness
	}

	ordering := rand.Perm(len(t.Population))
	crossOver := t.CrossOverSchedule.ValueAtTime(t.Generation)
	for i, j := range ordering[:len(ordering)-1] {
		if j < t.Elitism {
			continue
		}
		remainingIdxs := ordering[i+1:]
		otherIdx := remainingIdxs[rand.Intn(len(remainingIdxs))]
		keepRatio := 1 - crossOver
		e := t.Population[j]
		e1 := t.Population[otherIdx]
		e.Fitness = keepRatio*e.Fitness + (1-keepRatio)*e1.Fitness
		t.Crosser.Cross(e.Entity, e1.Entity, keepRatio)
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

	entities := make(chan *FitEntity, len(t.Population)-t.Elitism)
	for _, x := range t.Population[t.Elitism:] {
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
					e.Entity.Decay(decay)
				}
				t.Mutator.Mutate(t.Generation, e.Entity, gen)
			}
		}()
	}
	wg.Wait()
}

func (t *Trainer) reorderEntities() {
	if t.Elitism > 0 {
		s := fitnessSorter(t.Population)
		sort.Sort(s)
	}
	t.Selector.SetEntities(t.Population[t.Elitism:], t.FitnessScale())
	for i := t.Elitism; i < len(t.Population); i++ {
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
