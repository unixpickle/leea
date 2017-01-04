package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"os"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: analysis <network>")
		os.Exit(1)
	}
	data, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Read failed:", err)
		os.Exit(1)
	}
	net, err := neuralnet.DeserializeNetwork(data)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Deserialize failed:", err)
		os.Exit(1)
	}
	for i, p := range net.Parameters() {
		m1 := expectation(p.Vector)
		m2 := secondMoment(p.Vector)
		fmt.Printf("param %d: E[X]=%.2f\tE[X^2]=%.2f\tSigma(X)=%.2f\n", i, m1, m2,
			math.Sqrt(m2-m1*m1))
	}

	fmt.Println("Computing full error gradient...")
	g := neuralnet.BatchRGradienter{
		Learner:       net.BatchLearner(),
		CostFunc:      neuralnet.DotCost{},
		MaxBatchSize:  100,
		MaxGoroutines: 1,
	}
	grad := g.Gradient(mnist.LoadTrainingDataSet().SGDSampleSet())
	grad.Scale(1.0 / 60000.0)
	var vals []linalg.Vector
	for _, v := range grad {
		vals = append(vals, v)
	}
	fmt.Println("Gradient second moment:", secondMoment(vals...))
}

func expectation(x linalg.Vector) float64 {
	var sum float64
	for _, v := range x {
		sum += v
	}
	return sum / float64(len(x))
}

func secondMoment(x ...linalg.Vector) float64 {
	var sum float64
	var divisor float64
	for _, vec := range x {
		divisor += float64(len(vec))
		for _, v := range vec {
			sum += v * v
		}
	}
	return sum / divisor
}
