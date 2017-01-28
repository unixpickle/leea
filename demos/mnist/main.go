package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/leea"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/serializer"
)

var Creator anyvec.Creator

func main() {
	rand.Seed(time.Now().UnixNano())

	Creator = anyvec32.CurrentCreator()

	var mutInit, mutDecay, mutBaseline float64
	var crossInit, crossDecay, crossBaseline float64
	var decayTarget float64
	var inheritance float64
	var survivalRatio float64
	var population int
	var batchSize int
	var outFile string
	var convolutional bool
	var setMutations bool
	var elitism int
	var tournamentSize int
	var tournamentProb float64

	flag.Float64Var(&mutInit, "mut", 0.01, "mutation rate")
	flag.Float64Var(&mutDecay, "mutdecay", 0.999, "mutation decay rate")
	flag.Float64Var(&mutBaseline, "mutbias", 0.001, "mutation bias")

	flag.Float64Var(&decayTarget, "decay", 0.1, "decay target stddev")

	flag.Float64Var(&crossInit, "cross", 0.5, "cross-over rate")
	flag.Float64Var(&crossDecay, "crossdecay", 1, "cross-over decay rate")
	flag.Float64Var(&crossBaseline, "crossbias", 0, "cross-over bias")

	flag.Float64Var(&inheritance, "inherit", 0.95, "inheritance rate")
	flag.Float64Var(&survivalRatio, "survival", 0.2, "survival ratio")

	flag.IntVar(&tournamentSize, "tournsize", 5, "tournament size")
	flag.Float64Var(&tournamentProb, "tournprob", 1, "tournament probability")

	flag.IntVar(&population, "population", 512, "population size")
	flag.IntVar(&batchSize, "batch", 300, "samples per epoch")
	flag.IntVar(&elitism, "elitism", 0, "elite count")

	flag.StringVar(&outFile, "file", "out_net", "saved network file")
	flag.BoolVar(&convolutional, "conv", false, "use convolutional network")
	flag.BoolVar(&setMutations, "setmut", false, "use set mutations")

	flag.Parse()

	log.Println("Initializing trainer...")
	trainer := &leea.Trainer{
		Evaluator: &leea.NegCost{Cost: anynet.DotCost{}},
		Fetcher:   &anyff.Trainer{},
		Samples: &leea.CycleSampleSource{
			Samples:   mnist.LoadTrainingDataSet().AnyNetSamples(Creator),
			BatchSize: batchSize,
		},
		Selector: &leea.TournamentSelector{
			Size: tournamentSize,
			Prob: tournamentProb,
		},
		Crosser: &leea.NeuronalCrosser{},
		CrossOverSchedule: &leea.ExpSchedule{
			Init:      crossInit,
			DecayRate: crossDecay,
			Baseline:  crossBaseline,
		},
		Inheritance:   inheritance,
		SurvivalRatio: survivalRatio,
		Elitism:       elitism,
	}

	mutSchedule := &leea.ExpSchedule{
		Init:      mutInit,
		DecayRate: mutDecay,
		Baseline:  mutBaseline,
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
		trainer.Mutator = &leea.AddMutator{
			Stddev: mutSchedule,
		}
		trainer.DecaySchedule = &leea.DecaySchedule{
			Mut:    mutSchedule,
			Target: decayTarget,
		}
	}

	trainer.Population = populate(convolutional, outFile, population)

	log.Println("Training...")
	trainer.Evolve(func() bool {
		log.Printf("generation %d: max_fit=%f mean_fit=%f", trainer.Generation,
			trainer.MaxFitness()/trainer.FitnessScale(),
			trainer.MeanFitness()/trainer.FitnessScale())
		return true
	})

	log.Println("Saving fittest network...")
	net := trainer.BestEntity().Entity.(*leea.NetEntity).Parameterizer.(anynet.Net)
	if err := serializer.SaveAny(outFile, net); err != nil {
		log.Println("Save failed:", err)
	}

	crossValidate(net)
}

func populate(conv bool, outFile string, pop int) []*leea.FitEntity {
	var res []*leea.FitEntity

	var initNet anynet.Net
	err := serializer.LoadAny(outFile, &initNet)
	if err == nil {
		log.Println("Using existing network for population...")
	} else {
		log.Println("Creating population...")
	}
	for i := 0; i < pop; i++ {
		var net anynet.Layer
		if err == nil {
			data, _ := initNet.Serialize()
			net, _ = anynet.DeserializeNet(data)
		} else {
			if conv {
				net, err = anyconv.FromMarkup(Creator, convnetDescription)
				if err != nil {
					fmt.Fprintln(os.Stderr, "Create convnet:", err)
					os.Exit(1)
				}
			} else {
				net = anynet.Net{
					anynet.NewFC(Creator, 28*28, 300),
					anynet.Tanh,
					anynet.NewFC(Creator, 300, 10),
					anynet.LogSoftmax,
				}
			}
		}
		res = append(res, &leea.FitEntity{
			Entity: &leea.NetEntity{Parameterizer: net.(anynet.Parameterizer)},
		})
	}
	return res
}

func crossValidate(net anynet.Net) {
	log.Println("Cross-validating...")
	classif := func(s []float64) int {
		vec := Creator.MakeVectorData(Creator.MakeNumericList(s))
		out := net.Apply(anydiff.NewConst(vec), 1).Output()
		return anyvec.MaxIndex(out)
	}
	testSet := mnist.LoadTestingDataSet()
	total := testSet.NumCorrect(classif)
	log.Println("Total:", total)
	hist := testSet.CorrectnessHistogram(classif)
	log.Println(hist)
}

const convnetDescription = `
Input(w=28, h=28, d=1)
Conv(w=3, h=3, n=5)
MaxPool(w=3, h=3)
ReLU
FC(out=300)
ReLU
FC(out=10)
Softmax
`
