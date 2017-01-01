package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/leea"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

type Evaluator struct{}

func (_ Evaluator) Evaluate(e *leea.Entity, s sgd.SampleSet) float64 {
	b := e.Learner.(neuralnet.Network).BatchLearner()
	c := neuralnet.TotalCostBatcher(neuralnet.DotCost{}, b, s, 0)
	return -c / float64(s.Len())
}

func main() {
	var mutInit, mutDecay, mutBaseline float64
	var crossInit, crossDecay, crossBaseline float64
	var inheritance float64
	var survivalRatio float64
	var population int
	var batchSize int
	var outFile string

	flag.Float64Var(&mutInit, "mut", 1e-2, "mutation rate")
	flag.Float64Var(&mutDecay, "mutdecay", 0.999, "mutation decay rate")
	flag.Float64Var(&mutBaseline, "mutbias", 0, "mutation bias")

	flag.Float64Var(&crossInit, "cross", 1e-2, "cross-over rate")
	flag.Float64Var(&crossDecay, "crossdecay", 0.999, "cross-over decay rate")
	flag.Float64Var(&crossBaseline, "crossbias", 0, "cross-over bias")

	flag.Float64Var(&inheritance, "inherit", 0.99, "inheritance rate")
	flag.Float64Var(&survivalRatio, "survival", 0.2, "survival ratio")

	flag.IntVar(&population, "population", 256, "population size")
	flag.IntVar(&batchSize, "batch", 64, "samples per epoch")

	flag.StringVar(&outFile, "file", "out_net", "saved network file")

	flag.Parse()

	log.Println("Initializing trainer...")
	trainer := &leea.Trainer{
		Evaluator: Evaluator{},
		Samples: &leea.CycleSampleSource{
			Samples:   mnist.LoadTrainingDataSet().SGDSampleSet(),
			BatchSize: batchSize,
		},
		Selector: &leea.SortSelector{},
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

	netData, err := ioutil.ReadFile(outFile)
	if err == nil {
		log.Println("Using existing network for population...")
	} else {
		log.Println("Creating population...")
	}
	for i := 0; i < population; i++ {
		var net neuralnet.Network
		if err == nil {
			net, err = neuralnet.DeserializeNetwork(netData)
			if err != nil {
				fmt.Fprintln(os.Stderr, "Deserialize network:", err)
				os.Exit(1)
			}
		} else {
			net = neuralnet.Network{
				neuralnet.NewDenseLayer(28*28, 300),
				&neuralnet.HyperbolicTangent{},
				neuralnet.NewDenseLayer(300, 10),
				&neuralnet.SoftmaxLayer{},
			}
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

	log.Println("Saving fittest network...")
	net := trainer.BestEntity().Learner.(neuralnet.Network)
	netData, err = net.Serialize()
	if err != nil {
		log.Println("Serialize failed:", err)
	} else {
		if err := ioutil.WriteFile(outFile, netData, 0755); err != nil {
			log.Println("Save failed:", err)
		}
	}

	log.Println("Cross-validating...")
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
