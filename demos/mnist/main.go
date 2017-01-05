package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/leea"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/weakai/neuralnet"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var mutInit, mutDecay, mutBaseline float64
	var crossInit, crossDecay, crossBaseline float64
	var decayInit, decayDecay, decayBaseline float64
	var inheritance float64
	var survivalRatio float64
	var population int
	var batchSize int
	var outFile string
	var convolutional bool
	var hardEval bool
	var setMutations bool

	flag.Float64Var(&mutInit, "mut", 0.01, "mutation rate")
	flag.Float64Var(&mutDecay, "mutdecay", 0.996, "mutation decay rate")
	flag.Float64Var(&mutBaseline, "mutbias", 0.0001, "mutation bias")

	flag.Float64Var(&decayInit, "decay", 0, "weight decay rate")
	flag.Float64Var(&decayDecay, "decaydecay", 1, "weight decay decay rate")
	flag.Float64Var(&decayBaseline, "decaybias", 0, "weight decay bias")

	flag.Float64Var(&crossInit, "cross", 0.5, "cross-over rate")
	flag.Float64Var(&crossDecay, "crossdecay", 1, "cross-over decay rate")
	flag.Float64Var(&crossBaseline, "crossbias", 0, "cross-over bias")

	flag.Float64Var(&inheritance, "inherit", 0.95, "inheritance rate")
	flag.Float64Var(&survivalRatio, "survival", 0.2, "survival ratio")

	flag.IntVar(&population, "population", 512, "population size")
	flag.IntVar(&batchSize, "batch", 64, "samples per epoch")

	flag.StringVar(&outFile, "file", "out_net", "saved network file")
	flag.BoolVar(&convolutional, "conv", false, "use convolutional network")
	flag.BoolVar(&hardEval, "hard", false, "use % accuracy as fitness")
	flag.BoolVar(&setMutations, "setmut", false, "use set mutations")

	flag.Parse()

	log.Println("Initializing trainer...")
	mutSchedule := &leea.ExpSchedule{
		Init:      mutInit,
		DecayRate: mutDecay,
		Baseline:  mutBaseline,
	}
	trainer := &leea.Trainer{
		Evaluator: SoftEvaluator{},
		Samples: &leea.CycleSampleSource{
			Samples:   mnist.LoadTrainingDataSet().SGDSampleSet(),
			BatchSize: batchSize,
		},
		Selector: &leea.SortSelector{},
		Crosser:  &leea.NeuronalCrosser{},
		DecaySchedule: &leea.ExpSchedule{
			Init:      decayInit,
			DecayRate: decayDecay,
			Baseline:  decayBaseline,
		},
		CrossOverSchedule: &leea.ExpSchedule{
			Init:      crossInit,
			DecayRate: crossDecay,
			Baseline:  crossBaseline,
		},
		Inheritance:   inheritance,
		SurvivalRatio: survivalRatio,
	}

	if setMutations {
		log.Println("Using set mutations...")
		stddevs := []float64{0.10, 0.10, 0.10, 0.10}
		if convolutional {
			stddevs = []float64{0.10, 0.10, 0.10, 0.10, 0.10, 0.10}
		}
		trainer.Mutator = &leea.SetMutator{
			Stddevs:  stddevs,
			Fraction: mutSchedule,
		}
	} else {
		trainer.Mutator = &leea.AddMutator{Stddev: mutSchedule}
	}

	if hardEval {
		trainer.Evaluator = HardEvaluator{}
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
					neuralnet.NewDenseLayer(28*28, 300),
					&neuralnet.HyperbolicTangent{},
					neuralnet.NewDenseLayer(300, 10),
					&neuralnet.SoftmaxLayer{},
				}
			}
		}
		trainer.Population = append(trainer.Population, &leea.Entity{
			Learner: net,
		})
	}

	log.Println("Training...")
	trainer.Evolve(func() bool {
		log.Printf("generation %d: max_fit=%f", trainer.Generation,
			trainer.MaxFitness()/trainer.FitnessScale())
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
