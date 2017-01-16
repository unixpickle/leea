package main

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/leea"
)

type Crosser struct{}

func (_ Crosser) Cross(dest, source leea.Entity, keep float64) {
	if keep != 0.5 {
		panic("non-even keep probability is not supported")
	}

	destEnt := dest.(*Entity)
	sourceEnt := source.(*Entity)
	c := destEnt.RNN.Creator()

	for i, layer := range destEnt.RNN.Hidden {
		sourceLayer := sourceEnt.RNN.Hidden[i]
		choices := c.MakeVector(layer.StateSize)
		anyvec.Rand(choices, anyvec.Bernoulli, nil)
		maskedCrossover(layer.Biases, sourceLayer.Biases, choices)
		maskedCrossover(layer.InitState, sourceLayer.InitState, choices)
		maskedCrossover(layer.InTrans, sourceLayer.InTrans, choices)
		maskedCrossover(layer.StateTrans, sourceLayer.StateTrans, choices)
	}

	outChoices := c.MakeVector(destEnt.RNN.Output.OutSize)
	anyvec.Rand(outChoices, anyvec.Bernoulli, nil)
	maskedCrossover(destEnt.RNN.Output.Weights, sourceEnt.RNN.Output.Weights, outChoices)
	maskedCrossover(destEnt.RNN.Output.Biases, sourceEnt.RNN.Output.Biases, outChoices)
}

func maskedCrossover(v1, v2, choices anyvec.Vector) {
	choices = choices.Copy()
	v2 = v2.Copy()

	anyvec.ScaleChunks(v1, choices)
	choices.Scale(v1.Creator().MakeNumeric(-1))
	choices.AddScaler(v1.Creator().MakeNumeric(1))
	anyvec.ScaleChunks(v2, choices)
	v1.Add(v2)
}
