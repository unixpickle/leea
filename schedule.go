package leea

import "math"

// A Schedule determines how a parameter changes or decays
// over time.
type Schedule interface {
	ValueAtTime(timestep int) float64
}

// ExpSchedule exponentially decays a parameter over time.
// The parameter at time t is Baseline + Init*DecayRate^t
type ExpSchedule struct {
	Init      float64
	DecayRate float64
	Baseline  float64
}

// ValueAtTime returns the value after t timesteps of
// decay.
func (e *ExpSchedule) ValueAtTime(t int) float64 {
	return e.Baseline + e.Init*math.Pow(e.DecayRate, float64(t))
}

// A DecaySchedule tunes the decay parameter based on the
// mutation standard deviation in order to target a weight
// standard deviation.
type DecaySchedule struct {
	Mut    Schedule
	Target float64
}

// ValueAtTime chooses the right amount of decay to target
// the desired standard deviation, based on the current
// value of d.Mut.
func (d *DecaySchedule) ValueAtTime(t int) float64 {
	// Formula for final weight stddev was found empirically.
	// stddev ~= 0.8027 * noise * decay^-0.4646
	// decay ~= (stddev / (0.8027 * noise))^(-1/0.4646)
	noise := d.Mut.ValueAtTime(t)
	return math.Pow(d.Target/(0.8027*noise), -1/0.4646)
}
