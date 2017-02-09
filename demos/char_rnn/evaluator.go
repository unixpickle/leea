package main

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/leea"
)

type Evaluator struct{}

func (_ Evaluator) Evaluate(e leea.Entity, b anysgd.Batch) float64 {
	p := e.(*leea.NetEntity).Parameterizer
	block := p.(anyrnn.Block)
	tr := &anys2s.Trainer{
		Func: func(s anyseq.Seq) anyseq.Seq {
			s = anyseq.Map(s, func(v anydiff.Res, n int) anydiff.Res {
				c := v.Output().Creator()
				return anydiff.Scale(v, c.MakeNumeric(InputScale))
			})
			return anyrnn.Map(s, block)
		},
		Cost:    anynet.DotCost{},
		Params:  p.Parameters(),
		Average: true,
	}
	cost := anyvec.Sum(tr.TotalCost(b.(*anys2s.Batch)).Output())
	return -float64(cost.(float32))
}
