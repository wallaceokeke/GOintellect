package nn

import (
	"math"
	"math/rand"
	"time"

	"github.com/gointellect/gointellect/pkg/core"
)

// Layer interface defines the contract for neural network layers
type Layer interface {
	Forward(input *core.Tensor) *core.Tensor
	Backward(gradient *core.Tensor) *core.Tensor
	Parameters() []*core.Tensor
	SetParameters(params []*core.Tensor)
}

// DenseLayer represents a fully connected layer
type DenseLayer struct {
	Weights    *core.Tensor
	Bias       *core.Tensor
	InputSize  int
	OutputSize int
	Activation ActivationFunction
	LastInput  *core.Tensor
}

// NewDenseLayer creates a new dense layer
func NewDenseLayer(inputSize, outputSize int, activation ActivationFunction) *DenseLayer {
	// Initialize weights with Xavier/Glorot initialization
	limit := math.Sqrt(6.0 / float64(inputSize+outputSize))
	weights := core.Random([]int{inputSize, outputSize}).Scale(2 * limit).Add(core.Random([]int{inputSize, outputSize}).Scale(-limit))

	// Initialize bias to zero
	bias := core.Zeros([]int{outputSize})

	return &DenseLayer{
		Weights:    weights,
		Bias:       bias,
		InputSize:  inputSize,
		OutputSize: outputSize,
		Activation: activation,
	}
}

// Forward performs forward propagation
func (d *DenseLayer) Forward(input *core.Tensor) *core.Tensor {
	d.LastInput = input

	// Linear transformation: output = input * weights + bias
	linear := input.Dot(d.Weights)

	// Add bias (broadcasting)
	for i := 0; i < linear.Shape[0]; i++ {
		for j := 0; j < linear.Shape[1]; j++ {
			linear.Set(linear.At(i, j)+d.Bias.At(j), i, j)
		}
	}

	// Apply activation function
	return d.Activation.Forward(linear)
}

// Backward performs backward propagation
func (d *DenseLayer) Backward(gradient *core.Tensor) *core.Tensor {
	// Apply activation derivative
	gradient = d.Activation.Backward(gradient)

	// Compute gradients
	inputGradient := gradient.Dot(d.Weights.Transpose())
	weightGradient := d.LastInput.Transpose().Dot(gradient)

	// Update weights and bias (this would normally be done by optimizer)
	// For now, we'll just return the input gradient
	_ = weightGradient

	return inputGradient
}

// Parameters returns the layer's parameters
func (d *DenseLayer) Parameters() []*core.Tensor {
	return []*core.Tensor{d.Weights, d.Bias}
}

// SetParameters sets the layer's parameters
func (d *DenseLayer) SetParameters(params []*core.Tensor) {
	if len(params) != 2 {
		panic("DenseLayer requires exactly 2 parameters (weights, bias)")
	}
	d.Weights = params[0]
	d.Bias = params[1]
}

// ActivationFunction interface for activation functions
type ActivationFunction interface {
	Forward(input *core.Tensor) *core.Tensor
	Backward(gradient *core.Tensor) *core.Tensor
}

// ReLU activation function
type ReLU struct {
	LastInput *core.Tensor
}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (r *ReLU) Forward(input *core.Tensor) *core.Tensor {
	r.LastInput = input
	return input.Apply(func(x float64) float64 {
		if x > 0 {
			return x
		}
		return 0
	})
}

func (r *ReLU) Backward(gradient *core.Tensor) *core.Tensor {
	result := core.NewTensor(gradient.Shape)
	for i := 0; i < gradient.Size(); i++ {
		if r.LastInput.Data[i] > 0 {
			result.Data[i] = gradient.Data[i]
		} else {
			result.Data[i] = 0
		}
	}
	return result
}

// Sigmoid activation function
type Sigmoid struct {
	LastOutput *core.Tensor
}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (s *Sigmoid) Forward(input *core.Tensor) *core.Tensor {
	s.LastOutput = input.Apply(func(x float64) float64 {
		return 1.0 / (1.0 + math.Exp(-x))
	})
	return s.LastOutput
}

func (s *Sigmoid) Backward(gradient *core.Tensor) *core.Tensor {
	result := core.NewTensor(gradient.Shape)
	for i := 0; i < gradient.Size(); i++ {
		sig := s.LastOutput.Data[i]
		result.Data[i] = gradient.Data[i] * sig * (1 - sig)
	}
	return result
}

// Tanh activation function
type Tanh struct {
	LastOutput *core.Tensor
}

func NewTanh() *Tanh {
	return &Tanh{}
}

func (t *Tanh) Forward(input *core.Tensor) *core.Tensor {
	t.LastOutput = input.Apply(func(x float64) float64 {
		return math.Tanh(x)
	})
	return t.LastOutput
}

func (t *Tanh) Backward(gradient *core.Tensor) *core.Tensor {
	result := core.NewTensor(gradient.Shape)
	for i := 0; i < gradient.Size(); i++ {
		tanh := t.LastOutput.Data[i]
		result.Data[i] = gradient.Data[i] * (1 - tanh*tanh)
	}
	return result
}

// Softmax activation function
type Softmax struct {
	LastOutput *core.Tensor
}

func NewSoftmax() *Softmax {
	return &Softmax{}
}

func (s *Softmax) Forward(input *core.Tensor) *core.Tensor {
	// Apply softmax along the last dimension
	if input.Dims() != 2 {
		panic("Softmax requires 2D input")
	}

	result := core.NewTensor(input.Shape)

	for i := 0; i < input.Shape[0]; i++ {
		// Find max for numerical stability
		max := input.At(i, 0)
		for j := 1; j < input.Shape[1]; j++ {
			if input.At(i, j) > max {
				max = input.At(i, j)
			}
		}

		// Compute exponentials
		sum := 0.0
		for j := 0; j < input.Shape[1]; j++ {
			exp := math.Exp(input.At(i, j) - max)
			result.Set(exp, i, j)
			sum += exp
		}

		// Normalize
		for j := 0; j < input.Shape[1]; j++ {
			result.Set(result.At(i, j)/sum, i, j)
		}
	}

	s.LastOutput = result
	return result
}

func (s *Softmax) Backward(gradient *core.Tensor) *core.Tensor {
	// Softmax backward pass is complex and depends on the loss function
	// For now, return gradient as-is (this is simplified)
	return gradient
}

// NeuralNetwork represents a multi-layer neural network
type NeuralNetwork struct {
	Layers []Layer
}

// NewNeuralNetwork creates a new neural network
func NewNeuralNetwork(layers ...Layer) *NeuralNetwork {
	return &NeuralNetwork{
		Layers: layers,
	}
}

// Forward performs forward propagation through all layers
func (nn *NeuralNetwork) Forward(input *core.Tensor) *core.Tensor {
	output := input
	for _, layer := range nn.Layers {
		output = layer.Forward(output)
	}
	return output
}

// Backward performs backward propagation through all layers
func (nn *NeuralNetwork) Backward(gradient *core.Tensor) {
	for i := len(nn.Layers) - 1; i >= 0; i-- {
		gradient = nn.Layers[i].Backward(gradient)
	}
}

// Parameters returns all parameters from all layers
func (nn *NeuralNetwork) Parameters() []*core.Tensor {
	var params []*core.Tensor
	for _, layer := range nn.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}

// LossFunction interface for loss functions
type LossFunction interface {
	Forward(predictions, targets *core.Tensor) float64
	Backward(predictions, targets *core.Tensor) *core.Tensor
}

// MeanSquaredError loss function
type MeanSquaredError struct{}

func NewMeanSquaredError() *MeanSquaredError {
	return &MeanSquaredError{}
}

func (mse *MeanSquaredError) Forward(predictions, targets *core.Tensor) float64 {
	diff := predictions.Add(targets.Scale(-1))
	squared := diff.Mul(diff)
	return squared.Mean()
}

func (mse *MeanSquaredError) Backward(predictions, targets *core.Tensor) *core.Tensor {
	diff := predictions.Add(targets.Scale(-1))
	return diff.Scale(2.0 / float64(predictions.Size()))
}

// CrossEntropy loss function
type CrossEntropy struct{}

func NewCrossEntropy() *CrossEntropy {
	return &CrossEntropy{}
}

func (ce *CrossEntropy) Forward(predictions, targets *core.Tensor) float64 {
	// Add small epsilon to avoid log(0)
	epsilon := 1e-15
	loss := 0.0

	for i := 0; i < predictions.Size(); i++ {
		pred := math.Max(predictions.Data[i], epsilon)
		loss += -targets.Data[i] * math.Log(pred)
	}

	return loss / float64(predictions.Size())
}

func (ce *CrossEntropy) Backward(predictions, targets *core.Tensor) *core.Tensor {
	result := core.NewTensor(predictions.Shape)

	for i := 0; i < predictions.Size(); i++ {
		result.Data[i] = (predictions.Data[i] - targets.Data[i]) / float64(predictions.Size())
	}

	return result
}

// Optimizer interface for optimization algorithms
type Optimizer interface {
	Update(params []*core.Tensor, gradients []*core.Tensor)
}

// SGD optimizer
type SGD struct {
	LearningRate float64
}

func NewSGD(learningRate float64) *SGD {
	return &SGD{LearningRate: learningRate}
}

func (sgd *SGD) Update(params []*core.Tensor, gradients []*core.Tensor) {
	for i, param := range params {
		for j := range param.Data {
			param.Data[j] -= sgd.LearningRate * gradients[i].Data[j]
		}
	}
}

// Adam optimizer
type Adam struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
	M            []*core.Tensor // First moment estimates
	V            []*core.Tensor // Second moment estimates
	T            int            // Time step
}

func NewAdam(learningRate float64) *Adam {
	return &Adam{
		LearningRate: learningRate,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		T:            0,
	}
}

func (adam *Adam) Update(params []*core.Tensor, gradients []*core.Tensor) {
	// Initialize moment estimates if not done
	if len(adam.M) == 0 {
		adam.M = make([]*core.Tensor, len(params))
		adam.V = make([]*core.Tensor, len(params))
		for i, param := range params {
			adam.M[i] = core.Zeros(param.Shape)
			adam.V[i] = core.Zeros(param.Shape)
		}
	}

	adam.T++

	for i, param := range params {
		grad := gradients[i]

		// Update biased first moment estimate
		for j := range adam.M[i].Data {
			adam.M[i].Data[j] = adam.Beta1*adam.M[i].Data[j] + (1-adam.Beta1)*grad.Data[j]
		}

		// Update biased second moment estimate
		for j := range adam.V[i].Data {
			adam.V[i].Data[j] = adam.Beta2*adam.V[i].Data[j] + (1-adam.Beta2)*grad.Data[j]*grad.Data[j]
		}

		// Compute bias-corrected first moment estimate
		mHat := adam.M[i].Scale(1 / (1 - math.Pow(adam.Beta1, float64(adam.T))))

		// Compute bias-corrected second moment estimate
		vHat := adam.V[i].Scale(1 / (1 - math.Pow(adam.Beta2, float64(adam.T))))

		// Update parameters
		for j := range param.Data {
			param.Data[j] -= adam.LearningRate * mHat.Data[j] / (math.Sqrt(vHat.Data[j]) + adam.Epsilon)
		}
	}
}

// Initialize random seed
func init() {
	rand.Seed(time.Now().UnixNano())
}
