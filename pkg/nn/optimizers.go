package nn

import (
	"math"

	"github.com/wallaceokeke/GOintellect/pkg/core"
)

// RMSprop optimizer
type RMSprop struct {
	LearningRate float64
	Decay        float64
	Epsilon      float64
	Cache        []*core.Tensor
}

func NewRMSprop(learningRate float64) *RMSprop {
	return &RMSprop{
		LearningRate: learningRate,
		Decay:        0.9,
		Epsilon:      1e-8,
	}
}

func (r *RMSprop) Update(params []*core.Tensor, gradients []*core.Tensor) {
	// Initialize cache if not done
	if len(r.Cache) == 0 {
		r.Cache = make([]*core.Tensor, len(params))
		for i, param := range params {
			r.Cache[i] = core.Zeros(param.Shape)
		}
	}
	
	for i, param := range params {
		grad := gradients[i]
		
		// Update cache: cache = decay * cache + (1 - decay) * grad^2
		for j := range r.Cache[i].Data {
			r.Cache[i].Data[j] = r.Decay*r.Cache[i].Data[j] + (1-r.Decay)*grad.Data[j]*grad.Data[j]
		}
		
		// Update parameters: param = param - lr * grad / sqrt(cache + epsilon)
		for j := range param.Data {
			param.Data[j] -= r.LearningRate * grad.Data[j] / math.Sqrt(r.Cache[i].Data[j]+r.Epsilon)
		}
	}
}

// AdaGrad optimizer
type AdaGrad struct {
	LearningRate float64
	Epsilon      float64
	Cache        []*core.Tensor
}

func NewAdaGrad(learningRate float64) *AdaGrad {
	return &AdaGrad{
		LearningRate: learningRate,
		Epsilon:       1e-8,
	}
}

func (a *AdaGrad) Update(params []*core.Tensor, gradients []*core.Tensor) {
	// Initialize cache if not done
	if len(a.Cache) == 0 {
		a.Cache = make([]*core.Tensor, len(params))
		for i, param := range params {
			a.Cache[i] = core.Zeros(param.Shape)
		}
	}
	
	for i, param := range params {
		grad := gradients[i]
		
		// Update cache: cache = cache + grad^2
		for j := range a.Cache[i].Data {
			a.Cache[i].Data[j] += grad.Data[j] * grad.Data[j]
		}
		
		// Update parameters: param = param - lr * grad / sqrt(cache + epsilon)
		for j := range param.Data {
			param.Data[j] -= a.LearningRate * grad.Data[j] / math.Sqrt(a.Cache[i].Data[j]+a.Epsilon)
		}
	}
}

// AdaDelta optimizer
type AdaDelta struct {
	Decay        float64
	Epsilon      float64
	Cache        []*core.Tensor
	DeltaCache   []*core.Tensor
}

func NewAdaDelta() *AdaDelta {
	return &AdaDelta{
		Decay:   0.9,
		Epsilon: 1e-8,
	}
}

func (a *AdaDelta) Update(params []*core.Tensor, gradients []*core.Tensor) {
	// Initialize caches if not done
	if len(a.Cache) == 0 {
		a.Cache = make([]*core.Tensor, len(params))
		a.DeltaCache = make([]*core.Tensor, len(params))
		for i, param := range params {
			a.Cache[i] = core.Zeros(param.Shape)
			a.DeltaCache[i] = core.Zeros(param.Shape)
		}
	}
	
	for i, param := range params {
		grad := gradients[i]
		
		// Update cache: cache = decay * cache + (1 - decay) * grad^2
		for j := range a.Cache[i].Data {
			a.Cache[i].Data[j] = a.Decay*a.Cache[i].Data[j] + (1-a.Decay)*grad.Data[j]*grad.Data[j]
		}
		
		// Compute delta: delta = sqrt(delta_cache + epsilon) / sqrt(cache + epsilon) * grad
		for j := range param.Data {
			delta := math.Sqrt(a.DeltaCache[i].Data[j]+a.Epsilon) / math.Sqrt(a.Cache[i].Data[j]+a.Epsilon) * grad.Data[j]
			param.Data[j] -= delta
			
			// Update delta cache: delta_cache = decay * delta_cache + (1 - decay) * delta^2
			a.DeltaCache[i].Data[j] = a.Decay*a.DeltaCache[i].Data[j] + (1-a.Decay)*delta*delta
		}
	}
}

// Nadam optimizer (Adam with Nesterov momentum)
type Nadam struct {
	LearningRate float64
	Beta1         float64
	Beta2         float64
	Epsilon       float64
	M             []*core.Tensor // First moment estimates
	V             []*core.Tensor // Second moment estimates
	T             int            // Time step
}

func NewNadam(learningRate float64) *Nadam {
	return &Nadam{
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		T:            0,
	}
}

func (n *Nadam) Update(params []*core.Tensor, gradients []*core.Tensor) {
	// Initialize moment estimates if not done
	if len(n.M) == 0 {
		n.M = make([]*core.Tensor, len(params))
		n.V = make([]*core.Tensor, len(params))
		for i, param := range params {
			n.M[i] = core.Zeros(param.Shape)
			n.V[i] = core.Zeros(param.Shape)
		}
	}
	
	n.T++
	
	for i, param := range params {
		grad := gradients[i]
		
		// Update biased first moment estimate
		for j := range n.M[i].Data {
			n.M[i].Data[j] = n.Beta1*n.M[i].Data[j] + (1-n.Beta1)*grad.Data[j]
		}
		
		// Update biased second moment estimate
		for j := range n.V[i].Data {
			n.V[i].Data[j] = n.Beta2*n.V[i].Data[j] + (1-n.Beta2)*grad.Data[j]*grad.Data[j]
		}
		
		// Compute bias-corrected first moment estimate with Nesterov
		mHat := core.NewTensor(n.M[i].Shape)
		for j := range mHat.Data {
			mHat.Data[j] = (n.Beta1*n.M[i].Data[j] + (1-n.Beta1)*grad.Data[j]) / (1 - math.Pow(n.Beta1, float64(n.T)))
		}
		
		// Compute bias-corrected second moment estimate
		vHat := core.NewTensor(n.V[i].Shape)
		for j := range vHat.Data {
			vHat.Data[j] = n.V[i].Data[j] / (1 - math.Pow(n.Beta2, float64(n.T)))
		}
		
		// Update parameters
		for j := range param.Data {
			param.Data[j] -= n.LearningRate * mHat.Data[j] / (math.Sqrt(vHat.Data[j]) + n.Epsilon)
		}
	}
}

// L1Regularizer implements L1 regularization
type L1Regularizer struct {
	Lambda float64
}

func NewL1Regularizer(lambda float64) *L1Regularizer {
	return &L1Regularizer{Lambda: lambda}
}

func (l1 *L1Regularizer) Regularize(params []*core.Tensor) float64 {
	penalty := 0.0
	for _, param := range params {
		for _, val := range param.Data {
			penalty += math.Abs(val)
		}
	}
	return l1.Lambda * penalty
}

func (l1 *L1Regularizer) Gradient(params []*core.Tensor) []*core.Tensor {
	gradients := make([]*core.Tensor, len(params))
	for i, param := range params {
		gradients[i] = core.NewTensor(param.Shape)
		for j, val := range param.Data {
			if val > 0 {
				gradients[i].Data[j] = l1.Lambda
			} else if val < 0 {
				gradients[i].Data[j] = -l1.Lambda
			} else {
				gradients[i].Data[j] = 0
			}
		}
	}
	return gradients
}

// L2Regularizer implements L2 regularization
type L2Regularizer struct {
	Lambda float64
}

func NewL2Regularizer(lambda float64) *L2Regularizer {
	return &L2Regularizer{Lambda: lambda}
}

func (l2 *L2Regularizer) Regularize(params []*core.Tensor) float64 {
	penalty := 0.0
	for _, param := range params {
		for _, val := range param.Data {
			penalty += val * val
		}
	}
	return l2.Lambda * penalty / 2.0
}

func (l2 *L2Regularizer) Gradient(params []*core.Tensor) []*core.Tensor {
	gradients := make([]*core.Tensor, len(params))
	for i, param := range params {
		gradients[i] = core.NewTensor(param.Shape)
		for j, val := range param.Data {
			gradients[i].Data[j] = l2.Lambda * val
		}
	}
	return gradients
}

// ElasticNetRegularizer implements Elastic Net regularization
type ElasticNetRegularizer struct {
	L1Lambda float64
	L2Lambda float64
}

func NewElasticNetRegularizer(l1Lambda, l2Lambda float64) *ElasticNetRegularizer {
	return &ElasticNetRegularizer{
		L1Lambda: l1Lambda,
		L2Lambda: l2Lambda,
	}
}

func (en *ElasticNetRegularizer) Regularize(params []*core.Tensor) float64 {
	l1Penalty := 0.0
	l2Penalty := 0.0
	
	for _, param := range params {
		for _, val := range param.Data {
			l1Penalty += math.Abs(val)
			l2Penalty += val * val
		}
	}
	
	return en.L1Lambda*l1Penalty + en.L2Lambda*l2Penalty/2.0
}

func (en *ElasticNetRegularizer) Gradient(params []*core.Tensor) []*core.Tensor {
	gradients := make([]*core.Tensor, len(params))
	for i, param := range params {
		gradients[i] = core.NewTensor(param.Shape)
		for j, val := range param.Data {
			l1Grad := 0.0
			if val > 0 {
				l1Grad = en.L1Lambda
			} else if val < 0 {
				l1Grad = -en.L1Lambda
			}
			l2Grad := en.L2Lambda * val
			gradients[i].Data[j] = l1Grad + l2Grad
		}
	}
	return gradients
}

// LearningRateScheduler interface for learning rate scheduling
type LearningRateScheduler interface {
	GetLearningRate(epoch int) float64
}

// StepScheduler decreases learning rate by factor at specified epochs
type StepScheduler struct {
	InitialLR float64
	DecayRate float64
	StepSize  int
}

func NewStepScheduler(initialLR, decayRate float64, stepSize int) *StepScheduler {
	return &StepScheduler{
		InitialLR: initialLR,
		DecayRate: decayRate,
		StepSize:  stepSize,
	}
}

func (s *StepScheduler) GetLearningRate(epoch int) float64 {
	return s.InitialLR * math.Pow(s.DecayRate, float64(epoch/s.StepSize))
}

// ExponentialScheduler decreases learning rate exponentially
type ExponentialScheduler struct {
	InitialLR float64
	DecayRate float64
}

func NewExponentialScheduler(initialLR, decayRate float64) *ExponentialScheduler {
	return &ExponentialScheduler{
		InitialLR: initialLR,
		DecayRate: decayRate,
	}
}

func (e *ExponentialScheduler) GetLearningRate(epoch int) float64 {
	return e.InitialLR * math.Pow(e.DecayRate, float64(epoch))
}

// CosineAnnealingScheduler implements cosine annealing
type CosineAnnealingScheduler struct {
	InitialLR float64
	MinLR     float64
	TMax      int
}

func NewCosineAnnealingScheduler(initialLR, minLR float64, tMax int) *CosineAnnealingScheduler {
	return &CosineAnnealingScheduler{
		InitialLR: initialLR,
		MinLR:     minLR,
		TMax:      tMax,
	}
}

func (c *CosineAnnealingScheduler) GetLearningRate(epoch int) float64 {
	epoch = epoch % c.TMax
	return c.MinLR + (c.InitialLR-c.MinLR)*(1+math.Cos(math.Pi*float64(epoch)/float64(c.TMax)))/2
}

// ReduceLROnPlateauScheduler reduces learning rate when metric stops improving
type ReduceLROnPlateauScheduler struct {
	LearningRate float64
	Factor       float64
	Patience     int
	MinLR        float64
	BestScore    float64
	WaitCount    int
}

func NewReduceLROnPlateauScheduler(learningRate, factor, minLR float64, patience int) *ReduceLROnPlateauScheduler {
	return &ReduceLROnPlateauScheduler{
		LearningRate: learningRate,
		Factor:       factor,
		Patience:     patience,
		MinLR:        minLR,
		BestScore:    math.Inf(-1),
		WaitCount:    0,
	}
}

func (r *ReduceLROnPlateauScheduler) GetLearningRate(epoch int) float64 {
	return r.LearningRate
}

func (r *ReduceLROnPlateauScheduler) Update(score float64) {
	if score > r.BestScore {
		r.BestScore = score
		r.WaitCount = 0
	} else {
		r.WaitCount++
		if r.WaitCount >= r.Patience {
			r.LearningRate = math.Max(r.LearningRate*r.Factor, r.MinLR)
			r.WaitCount = 0
		}
	}
}
