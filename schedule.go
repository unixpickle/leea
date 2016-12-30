package leea

import "math"

// A Schedule determines how a parameter changes or decays
// over time.
type Schedule interface {
	ValueAtTime(timestep int) float64
}

// ExpSchedule exponentially decays a parameter over time.
// The parameter at time t is Init*DecayRate^t
type ExpSchedule struct {
	Init      float64
	DecayRate float64
}

// ValueAtTime returns the value after t timesteps of
// decay.
func (e *ExpSchedule) ValueAtTime(t int) float64 {
	return e.Init * math.Pow(e.DecayRate, float64(t))
}
