package leea

import (
	"math"
	"math/rand"
	"sort"
)

// A FitEntity is an entity-fitness pair.
type FitEntity struct {
	Entity  Entity
	Fitness float64
}

// A Selector chooses individuals based on their
// fitnesses.
type Selector interface {
	// SetEntities gives the Selector a set of entities to
	// work with.
	//
	// The receiver should not modify the slice or assume
	// that the slice will remain unchanged after the call.
	//
	// The scale indicates the value by which fitnesses
	// should be divided before being used.
	SetEntities(e []*FitEntity, scale float64)

	// Select selects the next entity.
	// Selection is done without replacement.
	Select() *FitEntity
}

// A RouletteWheel selects entities by randomly choosing
// them with probability proportional to their fitnesses.
//
// It is important to choose the proper units for the
// fitness measure.
// In particular, fitness values can never be negative.
type RouletteWheel struct {
	// Temperature controls how biased selections should be
	// towards more fit individuals.
	// If it is high (e.g. 10), then the fitnesses will
	// mostly be ignored.
	// If it is low (e.g. 0.1), then the fitnesses will
	// greatly influence selections.
	//
	// A Temperature of 0 is treated as 1, which uses the
	// exact fitnesses.
	Temperature float64

	entities []*FitEntity
	scale    float64
	total    float64
}

// SetEntities sets the entities for selection.
func (r *RouletteWheel) SetEntities(e []*FitEntity, scale float64) {
	r.entities = append([]*FitEntity{}, e...)
	r.scale = scale
	r.recomputeTotal()
}

// Select selects an entity and removes it from the pool.
func (r *RouletteWheel) Select() *FitEntity {
	num := rand.Float64() * r.total
	for i, e := range r.entities {
		num -= r.properFitness(e.Fitness)
		if i == len(r.entities)-1 || num < 0 {
			oldTotal := r.total
			r.total -= r.properFitness(e.Fitness)

			// Recompute if too much numerical precision was lost.
			if math.Abs(r.total/oldTotal) < 1e-3 {
				r.recomputeTotal()
			}

			r.entities[i] = r.entities[len(r.entities)-1]
			r.entities = r.entities[:len(r.entities)-1]
			return e
		}
	}
	panic("no entities to select")
}

func (r *RouletteWheel) properFitness(x float64) float64 {
	if x < 0 {
		panic("RouletteWheel requires non-negative fitnesses.")
	}
	if r.Temperature == 0 || r.Temperature == 1 {
		return x / r.scale
	}
	return math.Pow(x/r.scale, 1/r.Temperature)
}

func (r *RouletteWheel) recomputeTotal() {
	r.total = 0
	for _, e := range r.entities {
		r.total += r.properFitness(e.Fitness)
	}
}

// A SortSelector selects entities in order of their
// fitnesses.
type SortSelector struct {
	entities []*FitEntity
}

// SetEntities sets the entities for selection.
func (s *SortSelector) SetEntities(e []*FitEntity, scale float64) {
	x := append(fitnessSorter{}, e...)
	sort.Sort(x)
	s.entities = x
}

// Select selects an entity and removes it from the pool.
func (s *SortSelector) Select() *FitEntity {
	if len(s.entities) == 0 {
		panic("no entities to select")
	}
	res := s.entities[0]
	s.entities = s.entities[1:]
	return res
}

// TournamentSelector uses the tournament selection method
// to select entities based on fitness.
type TournamentSelector struct {
	// Size is the tournament size.
	Size int

	// Prob is the probability of selecting the top entity
	// in a tournament.
	// The lower Prob is, the less likely the most fit entity
	// is to be chosen.
	Prob float64

	entities []*FitEntity
}

// SetEntities sets the entities for selection.
func (t *TournamentSelector) SetEntities(e []*FitEntity, scale float64) {
	t.entities = append([]*FitEntity{}, e...)
}

// Select selects an entity and removes it.
func (t *TournamentSelector) Select() *FitEntity {
	pool := t.tournamentPool()
	prob := t.Prob
	var chosen *FitEntity
	for i, entry := range pool {
		if rand.Float64() < prob || i == len(pool)-1 {
			chosen = entry
			break
		}
	}

	for i, x := range t.entities {
		if x == chosen {
			t.entities[i] = t.entities[len(t.entities)-1]
			t.entities = t.entities[:len(t.entities)-1]
			break
		}
	}

	return chosen
}

func (t *TournamentSelector) tournamentPool() []*FitEntity {
	var s fitnessSorter
	if len(t.entities) < t.Size {
		s = append(s, t.entities...)
	} else {
		indices := rand.Perm(len(t.entities))
		for _, j := range indices[:t.Size] {
			s = append(s, t.entities[j])
		}
	}
	sort.Sort(s)
	return s
}

type fitnessSorter []*FitEntity

func (f fitnessSorter) Len() int {
	return len(f)
}

func (f fitnessSorter) Swap(i, j int) {
	f[i], f[j] = f[j], f[i]
}

func (f fitnessSorter) Less(i, j int) bool {
	return f[i].Fitness > f[j].Fitness
}
