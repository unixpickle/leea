package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math"
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
	var convolutional bool

	flag.Float64Var(&mutInit, "mut", 5e-3, "mutation rate")
	flag.Float64Var(&mutDecay, "mutdecay", 1, "mutation decay rate")
	flag.Float64Var(&mutBaseline, "mutbias", 0, "mutation bias")

	flag.Float64Var(&crossInit, "cross", 0.5, "cross-over rate")
	flag.Float64Var(&crossDecay, "crossdecay", 0.999, "cross-over decay rate")
	flag.Float64Var(&crossBaseline, "crossbias", 0, "cross-over bias")

	flag.Float64Var(&inheritance, "inherit", 0.95, "inheritance rate")
	flag.Float64Var(&survivalRatio, "survival", 0.2, "survival ratio")

	flag.IntVar(&population, "population", 512, "population size")
	flag.IntVar(&batchSize, "batch", 64, "samples per epoch")

	flag.StringVar(&outFile, "file", "out_net", "saved network file")
	flag.BoolVar(&convolutional, "conv", false, "use convolutional network")

	flag.Parse()

	log.Println("Initializing trainer...")
	trainer := &leea.Trainer{
		Evaluator: Evaluator{},
		Samples: &leea.CycleSampleSource{
			Samples:   mnist.LoadTrainingDataSet().SGDSampleSet(),
			BatchSize: batchSize,
		},
		Selector: &leea.SortSelector{},
		Crosser:  &leea.NeuronalCrosser{},
		Mutator:  &leea.NormMutator{MagDrift: 0.01},
		MutationSchedule: &leea.ExpSchedule{
			Init:      mutInit,
			DecayRate: mutDecay,
			Baseline:  mutBaseline,
		},
		CrossOverSchedule: &leea.ExpSchedule{
			Init:      crossInit,
			DecayRate: crossDecay,
			Baseline:  crossBaseline,
		},
		Inheritance:   inheritance,
		SurvivalRatio: survivalRatio,
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
			if convolutional {
				net = createConvNet()
			} else {
				net = neuralnet.Network{
					&neuralnet.RescaleLayer{Scale: 3 / math.Sqrt(28*28)},
					neuralnet.NewDenseLayer(28*28, 300),
					&neuralnet.HyperbolicTangent{},
					&neuralnet.RescaleLayer{Scale: 3 / math.Sqrt(300)},
					neuralnet.NewDenseLayer(300, 10),
					&neuralnet.SoftmaxLayer{},
				}
				net[1].(*neuralnet.DenseLayer).Weights.Data.Vector.Scale(math.Sqrt(28 * 28))
				net[4].(*neuralnet.DenseLayer).Weights.Data.Vector.Scale(math.Sqrt(300))
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

func createConvNet() neuralnet.Network {
	const (
		HiddenSize     = 300
		FilterSize     = 3
		FilterCount    = 5
		FilterStride   = 1
		MaxPoolingSpan = 3
	)

	convOutWidth := (28-FilterSize)/FilterStride + 1
	convOutHeight := (28-FilterSize)/FilterStride + 1

	poolOutWidth := convOutWidth / MaxPoolingSpan
	if convOutWidth%MaxPoolingSpan != 0 {
		poolOutWidth++
	}
	poolOutHeight := convOutWidth / MaxPoolingSpan
	if convOutHeight%MaxPoolingSpan != 0 {
		poolOutHeight++
	}
	net := neuralnet.Network{
		&neuralnet.ConvLayer{
			FilterCount:  FilterCount,
			FilterWidth:  FilterSize,
			FilterHeight: FilterSize,
			Stride:       FilterStride,
			InputWidth:   28,
			InputHeight:  28,
			InputDepth:   1,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.MaxPoolingLayer{
			XSpan:       MaxPoolingSpan,
			YSpan:       MaxPoolingSpan,
			InputWidth:  convOutWidth,
			InputHeight: convOutHeight,
			InputDepth:  FilterCount,
		},
		&neuralnet.DenseLayer{
			InputCount:  poolOutWidth * poolOutHeight * FilterCount,
			OutputCount: HiddenSize,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  HiddenSize,
			OutputCount: 10,
		},
		&neuralnet.SoftmaxLayer{},
	}
	net.Randomize()
	return net
}
