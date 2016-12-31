package leea

import (
	"math/rand"
	"sort"
)

// A Selector chooses individuals based on their
// fitnesses.
type Selector interface {
	// SetEntities gives the Selector a set of entities to
	// work with.
	//
	// The receiver should not modify the slice or assume
	// that the slice will remain unchanged after the call.
	SetEntities(e []*Entity)

	// Select selects the next entity.
	// Selection is done without replacement.
	Select() *Entity
}

// A RouletteWheel selects entities by randomly choosing
// them with probability proportional to their fitnesses.
type RouletteWheel struct {
	entities []*Entity
	total    float64
}

// SetEntities sets the entities for selection.
func (r *RouletteWheel) SetEntities(e []*Entity) {
	r.total = 0
	for _, e := range e {
		r.total += e.Fitness
	}
	r.entities = append([]*Entity{}, e...)
}

// Select selects an entity and removes it from the pool.
func (r *RouletteWheel) Select() *Entity {
	num := rand.Float64() * r.total
	for i, e := range r.entities {
		num -= e.Fitness
		if i == len(r.entities)-1 || num < 0 {
			r.total -= e.Fitness
			r.entities[i] = r.entities[len(r.entities)-1]
			r.entities = r.entities[:len(r.entities)-1]
			return e
		}
	}
	panic("no entities to select")
}

// A SortSelector selects entities in order of their
// fitnesses.
type SortSelector struct {
	entities []*Entity
}

// SetEntities sets the entities for selection.
func (s *SortSelector) SetEntities(e []*Entity) {
	x := append(fitnessSorter{}, e...)
	sort.Sort(x)
	s.entities = x
}

// Select selects an entity and removes it from the pool.
func (s *SortSelector) Select() *Entity {
	if len(s.entities) == 0 {
		panic("no entities to select")
	}
	res := s.entities[0]
	s.entities = s.entities[1:]
	return res
}

type fitnessSorter []*Entity

func (f fitnessSorter) Len() int {
	return len(f)
}

func (f fitnessSorter) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}

func (f fitnessSorter) Less(i, j int) bool {
	return f[i].Fitness > f[j].Fitness
}
