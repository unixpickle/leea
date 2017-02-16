package main

import (
	"fmt"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/mnist"
)

func main() {
	c := anyvec32.CurrentCreator()
	net := anynet.Net{
		anynet.NewFC(c, 28*28, 300),
		anynet.Tanh,
		anynet.NewFC(c, 300, 10),
		anynet.LogSoftmax,
	}

	tr := &anyff.Trainer{
		Net:     net,
		Cost:    anynet.DotCost{},
		Params:  net.Parameters(),
		Average: true,
	}
	testSet := mnist.LoadTestingDataSet()
	var iter int
	sgd := &anysgd.SGD{
		Fetcher:    tr,
		Gradienter: tr,
		BatchSize:  300,
		Rater:      anysgd.ConstRater(0.01),
		Samples:    mnist.LoadTrainingDataSet().AnyNetSamples(c),
		StatusFunc: func(b anysgd.Batch) {
			cf := func(in []float64) int {
				inVec := c.MakeVectorData(c.MakeNumericList(in))
				out := net.Apply(anydiff.NewConst(inVec), 1)
				return anyvec.MaxIndex(out.Output())
			}
			fmt.Printf("%f,%f\n", float64(iter)/60000,
				float64(testSet.NumCorrect(cf))/10000)
			iter += 300
		},
	}
	sgd.Run(nil)
}
