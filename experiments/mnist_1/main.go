package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/leea"
	"github.com/unixpickle/mnist"
)

var Creator anyvec.Creator

const (
	BatchSize  = 300
	Population = 512
)

func main() {
	Creator = anyvec32.CurrentCreator()

	flag.Parse()

	log.Println("Initializing trainer...")
	mutSchedule := &leea.ExpSchedule{
		Init:      0.01,
		DecayRate: 0.999,
		Baseline:  0.001,
	}
	trainer := &leea.Trainer{
		Evaluator: &leea.NegCost{Cost: anynet.DotCost{}},
		Fetcher:   &anyff.Trainer{},
		Samples: &leea.CycleSampleSource{
			Samples:   mnist.LoadTrainingDataSet().AnyNetSamples(Creator),
			BatchSize: 300,
		},
		Selector:          &leea.TournamentSelector{Size: 5, Prob: 1},
		Crosser:           &leea.NeuronalCrosser{},
		CrossOverSchedule: &leea.ExpSchedule{Baseline: 0.5},
		Inheritance:       0.95,
		SurvivalRatio:     0.2,
		Elitism:           0,
		Mutator:           &leea.AddMutator{Stddev: mutSchedule},
		DecaySchedule: &leea.DecaySchedule{
			Mut:    mutSchedule,
			Target: 0.1,
		},
	}

	trainer.Population = populate(Population)

	log.Println("Training...")
	trainer.Evolve(func() bool {
		log.Printf("generation %d: max_fit=%f mean_fit=%f", trainer.Generation,
			trainer.MaxFitness()/trainer.FitnessScale(),
			trainer.MeanFitness()/trainer.FitnessScale())
		if trainer.Generation%10 == 0 {
			entity := trainer.BestEntity().Entity.(*leea.NetEntity)
			accuracy := crossValidate(entity.Parameterizer.(anynet.Net))
			numSamples := trainer.Generation * BatchSize
			fmt.Printf("%f,%f,%f\n", float64(numSamples)/60000, accuracy,
				trainer.MaxFitness()/trainer.FitnessScale())
		}
		return true
	})
}

func populate(pop int) []*leea.FitEntity {
	var res []*leea.FitEntity

	log.Println("Populating...")
	for i := 0; i < pop; i++ {
		net := anynet.Net{
			anynet.NewFC(Creator, 28*28, 300),
			anynet.Tanh,
			anynet.NewFC(Creator, 300, 10),
			anynet.LogSoftmax,
		}
		res = append(res, &leea.FitEntity{
			Entity: &leea.NetEntity{Parameterizer: net},
		})
	}

	return res
}

func crossValidate(net anynet.Net) float64 {
	log.Println("Cross-validating...")
	classif := func(s []float64) int {
		vec := Creator.MakeVectorData(Creator.MakeNumericList(s))
		out := net.Apply(anydiff.NewConst(vec), 1).Output()
		return anyvec.MaxIndex(out)
	}
	testSet := mnist.LoadTestingDataSet()
	total := testSet.NumCorrect(classif)
	return float64(total) / float64(len(testSet.Samples))
}
