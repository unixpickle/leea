package leea

import "testing"

func TestRouletteWheel(t *testing.T) {
	selector := &RouletteWheel{Temperature: 0.05}
	for i := 0; i < 10; i++ {
		selector.SetEntities([]*Entity{
			{Fitness: 1},
			{Fitness: 1.5},
			{Fitness: 0.5},
			{Fitness: 10},
			{Fitness: 10},
			{Fitness: 100},
			{Fitness: 1000},
		})
		var fits []float64
		for i := 0; i < 4; i++ {
			fits = append(fits, selector.Select().Fitness)
		}
		if fits[0] != 1000 || fits[1] != 100 || fits[2] != 10 || fits[3] != 10 {
			t.Errorf("expected [1000 100 10 10] but got %v", fits)
		}
	}
}
