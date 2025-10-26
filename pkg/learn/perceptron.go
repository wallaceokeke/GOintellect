package learn

// Perceptron - simple binary classifier
type Perceptron struct {
    Weights []float64
    Bias    float64
    LR      float64
}

func NewPerceptron(n int, lr float64) *Perceptron {
    return &Perceptron{
        Weights: make([]float64, n),
        Bias:    0,
        LR:      lr,
    }
}

func (p *Perceptron) Predict(x []float64) int {
    var sum float64
    for i := range p.Weights {
        sum += p.Weights[i] * x[i]
    }
    sum += p.Bias
    if sum >= 0 {
        return 1
    }
    return 0
}

func (p *Perceptron) Train(X [][]float64, Y []int, epochs int) {
    for e := 0; e < epochs; e++ {
        for i, x := range X {
            pred := p.Predict(x)
            err := Y[i] - pred
            if err != 0 {
                for j := range p.Weights {
                    p.Weights[j] += p.LR * float64(err) * x[j]
                }
                p.Bias += p.LR * float64(err)
            }
        }
    }
}
