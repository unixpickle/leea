package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/leea"
	"github.com/unixpickle/serializer"
)

const (
	StateSize  = 512
	EndSamples = 5
)

func main() {
	var mutInit, mutDecay, mutBaseline float64
	var decayTarget float64
	var inheritance float64
	var survivalRatio float64
	var population int
	var batchSize int
	var dataFile, outFile string
	var tournamentSize int
	var tournamentProb float64

	flag.Float64Var(&mutInit, "mut", 1e-2, "mutation rate")
	flag.Float64Var(&mutDecay, "mutdecay", 0.999, "mutation decay rate")
	flag.Float64Var(&mutBaseline, "mutbias", 0.001, "mutation bias")

	flag.Float64Var(&decayTarget, "decay", 0.05, "decay target")

	flag.IntVar(&tournamentSize, "tournsize", 5, "tournament size")
	flag.Float64Var(&tournamentProb, "tournprob", 1, "tournament probability")

	flag.Float64Var(&inheritance, "inherit", 0.95, "inheritance rate")
	flag.Float64Var(&survivalRatio, "survival", 0.2, "survival ratio")

	flag.IntVar(&population, "population", 16, "population size")
	flag.IntVar(&batchSize, "batch", 64, "samples per epoch")

	flag.StringVar(&dataFile, "data", "", "text data file")
	flag.StringVar(&outFile, "file", "out_net", "saved network file")

	flag.Parse()

	if dataFile == "" {
		fmt.Fprintln(os.Stderr, "Must specify -data flag. See -help for more.")
		os.Exit(1)
	}

	log.Println("Reading training data...")
	samples, err := ReadSamples(dataFile)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load samples:", err)
		os.Exit(1)
	}

	log.Println("Initializing trainer...")
	mutSchedule := &leea.ExpSchedule{
		Init:      mutInit,
		DecayRate: mutDecay,
		Baseline:  mutBaseline,
	}
	trainer := &leea.Trainer{
		Fetcher:   &anys2s.Trainer{},
		Evaluator: Evaluator{},
		Samples: &leea.CycleSampleSource{
			Samples:   samples,
			BatchSize: batchSize,
		},
		Selector: &leea.TournamentSelector{
			Size: tournamentSize,
			Prob: tournamentProb,
		},
		Crosser: &leea.NeuronalCrosser{},
		Mutator: &leea.AddMutator{
			Stddev: mutSchedule,
		},
		DecaySchedule: &leea.DecaySchedule{
			Mut:    mutSchedule,
			Target: decayTarget,
		},
		CrossOverSchedule: &leea.ExpSchedule{
			Baseline: 0.5,
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
		var net anyrnn.Stack
		if err == nil {
			err = serializer.DeserializeAny(netData, &net)
			if err != nil {
				essentials.Die("Deserialize network:", err)
			}
		} else {
			c := anyvec32.CurrentCreator()
			net = anyrnn.Stack{
				anyrnn.NewVanillaZero(c, 0x100, StateSize, anynet.Tanh),
				anyrnn.NewVanillaZero(c, StateSize, StateSize, anynet.Tanh),
				&anyrnn.LayerBlock{
					Layer: anynet.Net{
						anynet.NewFCZero(c, StateSize, 0x100),
						anynet.LogSoftmax,
					},
				},
			}
		}
		trainer.Population = append(trainer.Population, &leea.FitEntity{
			Entity: &leea.NetEntity{Parameterizer: net},
		})
	}

	log.Println("Training...")
	trainer.Evolve(func() bool {
		log.Printf("generation %d: max_fit=%f", trainer.Generation,
			trainer.MaxFitness()/trainer.FitnessScale())
		return true
	})

	log.Println("Saving fittest network...")
	net := trainer.BestEntity().Entity.(*leea.NetEntity).Parameterizer.(anyrnn.Block)
	if err := serializer.SaveAny(outFile, net); err != nil {
		log.Println("Save failed:", err)
	}

	log.Println("Producing samples...")
	for i := 0; i < EndSamples; i++ {
		fmt.Println(GenerateSample(net))
	}
}
