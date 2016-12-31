package main

import (
	"log"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/leea"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	PopSize   = 256
	BatchSize = 64
)

type Evaluator struct{}

func (_ Evaluator) Evaluate(e *leea.Entity, s sgd.SampleSet) float64 {
	b := e.Learner.(neuralnet.Network).BatchLearner()
	c := neuralnet.TotalCostBatcher(neuralnet.DotCost{}, b, s, 0)
	return -c / float64(s.Len())
}

func main() {
	log.Println("Initializing trainer...")
	trainer := &leea.Trainer{
		Evaluator: Evaluator{},
		Samples: &leea.CycleSampleSource{
			Samples:   mnist.LoadTrainingDataSet().SGDSampleSet(),
			BatchSize: BatchSize,
		},
		Selector: &leea.RouletteWheel{Temperature: 0.05},
		MutationSchedule: &leea.ExpSchedule{
			Init:      1e-2,
			DecayRate: 0.999,
		},
		CrossOverSchedule: &leea.ExpSchedule{
			Init:      0.01,
			DecayRate: 0.999,
		},
		Inheritance:   0.99,
		SurvivalRatio: 0.2,
	}

	log.Println("Creating population...")
	for i := 0; i < PopSize; i++ {
		net := neuralnet.Network{
			neuralnet.NewDenseLayer(28*28, 300),
			&neuralnet.HyperbolicTangent{},
			neuralnet.NewDenseLayer(300, 10),
			&neuralnet.SoftmaxLayer{},
		}
		trainer.Population = append(trainer.Population, &leea.Entity{
			Learner: net,
		})
	}

	log.Println("Training...")
	fitScale := 1.0
	trainer.Evolve(func() bool {
		log.Printf("generation %d: max_fit=%f", trainer.Generation,
			trainer.MaxFitness()/fitScale)
		fitScale *= trainer.Inheritance
		fitScale += 1
		return true
	})

	log.Println("Cross-validating...")
	net := trainer.BestEntity().Learner.(neuralnet.Network)
	classif := func(s []float64) int {
		out := net.Apply(&autofunc.Variable{Vector: s}).Output()
		_, res := out.Max()
		return res
	}
	total := mnist.LoadTestingDataSet().NumCorrect(classif)
	log.Println("Total:", total)
	hist := mnist.LoadTestingDataSet().CorrectnessHistogram(classif)
	log.Println(hist)
}
