package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/unixpickle/leea"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	StateSize  = 300
	EndSamples = 5
)

func main() {
	var mutInit, mutDecay, mutBaseline float64
	var crossInit, crossDecay, crossBaseline float64
	var inheritance float64
	var survivalRatio float64
	var population int
	var batchSize int
	var dataFile, outFile string

	flag.Float64Var(&mutInit, "mut", 1e-2, "mutation rate")
	flag.Float64Var(&mutDecay, "mutdecay", 0.999, "mutation decay rate")
	flag.Float64Var(&mutBaseline, "mutbias", 0, "mutation bias")

	flag.Float64Var(&crossInit, "cross", 0.5, "cross-over rate")
	flag.Float64Var(&crossDecay, "crossdecay", 1, "cross-over decay rate")
	flag.Float64Var(&crossBaseline, "crossbias", 0, "cross-over bias")

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
	trainer := &leea.Trainer{
		Evaluator: Evaluator{},
		Samples: &leea.CycleSampleSource{
			Samples:   samples,
			BatchSize: batchSize,
		},
		Selector: &leea.SortSelector{},
		Crosser:  &leea.NeuronalCrosser{},
		Mutator: &leea.SetMutator{
			Fraction: &leea.ExpSchedule{
				Init:      mutInit,
				DecayRate: mutDecay,
				Baseline:  mutBaseline,
			},
			Stddevs: []float64{0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
				0.05, 0.05, 0.05},
			Sampler: &leea.NormSampler{},
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
		var net rnn.StackedBlock
		if err == nil {
			net, err = rnn.DeserializeStackedBlock(netData)
			if err != nil {
				fmt.Fprintln(os.Stderr, "Deserialize network:", err)
				os.Exit(1)
			}
		} else {
			net = rnn.StackedBlock{
				&rnn.StateOutBlock{
					Block: rnn.NewNetworkBlock(neuralnet.Network{
						neuralnet.NewDenseLayer(StateSize+0x100, StateSize),
						&neuralnet.HyperbolicTangent{},
					}, StateSize),
				},
				&rnn.StateOutBlock{
					Block: rnn.NewNetworkBlock(neuralnet.Network{
						neuralnet.NewDenseLayer(StateSize*2, StateSize),
						&neuralnet.HyperbolicTangent{},
					}, StateSize),
				},
				rnn.NewNetworkBlock(neuralnet.Network{
					neuralnet.NewDenseLayer(StateSize, 0x100),
					&neuralnet.LogSoftmaxLayer{},
				}, 0),
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
	net := trainer.BestEntity().Learner.(rnn.StackedBlock)
	netData, err = net.Serialize()
	if err != nil {
		log.Println("Serialize failed:", err)
	} else {
		if err := ioutil.WriteFile(outFile, netData, 0755); err != nil {
			log.Println("Save failed:", err)
		}
	}

	log.Println("Producing samples...")
	for i := 0; i < EndSamples; i++ {
		fmt.Println(GenerateSample(net))
	}
}
