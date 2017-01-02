package main

import (
	"io/ioutil"
	"log"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

func main() {
	net := neuralnet.Network{
		neuralnet.NewDenseLayer(28*28, 300),
		&neuralnet.HyperbolicTangent{},
		neuralnet.NewDenseLayer(300, 10),
		&neuralnet.LogSoftmaxLayer{},
	}
	net[0].(*neuralnet.DenseLayer).Biases.Var.Vector.Scale(0)
	net[2].(*neuralnet.DenseLayer).Biases.Var.Vector.Scale(0)
	g := &sgd.Adam{
		Gradienter: &neuralnet.BatchRGradienter{
			Learner:  net.BatchLearner(),
			CostFunc: neuralnet.DotCost{},
		},
	}
	log.Println("Training...")
	sgd.SGD(g, mnist.LoadTrainingDataSet().SGDSampleSet(), 0.001, 1, 100)
	log.Println("Testing...")
	ts := mnist.LoadTestingDataSet()
	cf := func(in []float64) int {
		res := net.Apply(&autofunc.Variable{Vector: in}).Output()
		_, max := res.Max()
		return max
	}
	log.Println("Validation:", ts.NumCorrect(cf))
	log.Println("Histogram:", ts.CorrectnessHistogram(cf))

	data, _ := net.Serialize()
	ioutil.WriteFile("out_net", data, 0755)
}
