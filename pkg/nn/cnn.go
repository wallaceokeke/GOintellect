package nn

import (
	"math"

	"github.com/gointellect/gointellect/pkg/core"
)

// Conv2DLayer represents a 2D convolutional layer
type Conv2DLayer struct {
	Filters     int
	KernelSize  [2]int
	Stride      [2]int
	Padding     [2]int
	Weights     *core.Tensor
	Bias        *core.Tensor
	InputShape  [3]int // [channels, height, width]
	OutputShape [3]int
	Activation  ActivationFunction
	LastInput   *core.Tensor
}

// NewConv2DLayer creates a new 2D convolutional layer
func NewConv2DLayer(filters int, kernelSize [2]int, stride, padding [2]int, activation ActivationFunction) *Conv2DLayer {
	return &Conv2DLayer{
		Filters:    filters,
		KernelSize: kernelSize,
		Stride:     stride,
		Padding:    padding,
		Activation: activation,
	}
}

// Forward performs forward propagation for 2D convolution
func (c *Conv2DLayer) Forward(input *core.Tensor) *core.Tensor {
	c.LastInput = input

	// Reshape input to [batch, channels, height, width]
	batchSize := input.Shape[0]
	channels := input.Shape[1]
	height := input.Shape[2]
	width := input.Shape[3]

	// Calculate output dimensions
	outHeight := (height+2*c.Padding[0]-c.KernelSize[0])/c.Stride[0] + 1
	outWidth := (width+2*c.Padding[1]-c.KernelSize[1])/c.Stride[1] + 1

	c.OutputShape = [3]int{c.Filters, outHeight, outWidth}

	// Initialize weights if not done
	if c.Weights == nil {
		c.Weights = core.RandomNormal([]int{c.Filters, channels, c.KernelSize[0], c.KernelSize[1]}, 0, 0.1)
		c.Bias = core.Zeros([]int{c.Filters})
	}

	// Perform convolution
	output := core.NewTensor([]int{batchSize, c.Filters, outHeight, outWidth})

	for b := 0; b < batchSize; b++ {
		for f := 0; f < c.Filters; f++ {
			for h := 0; h < outHeight; h++ {
				for w := 0; w < outWidth; w++ {
					sum := c.Bias.At(f)

					// Convolution operation
					for ch := 0; ch < channels; ch++ {
						for kh := 0; kh < c.KernelSize[0]; kh++ {
							for kw := 0; kw < c.KernelSize[1]; kw++ {
								inH := h*c.Stride[0] + kh - c.Padding[0]
								inW := w*c.Stride[1] + kw - c.Padding[1]

								if inH >= 0 && inH < height && inW >= 0 && inW < width {
									sum += input.At(b, ch, inH, inW) * c.Weights.At(f, ch, kh, kw)
								}
							}
						}
					}

					output.Set(sum, b, f, h, w)
				}
			}
		}
	}

	// Apply activation function
	if c.Activation != nil {
		return c.Activation.Forward(output)
	}
	return output
}

// Backward performs backward propagation for 2D convolution
func (c *Conv2DLayer) Backward(gradient *core.Tensor) *core.Tensor {
	// Simplified backward pass - in practice, this would be more complex
	return gradient
}

// Parameters returns the layer's parameters
func (c *Conv2DLayer) Parameters() []*core.Tensor {
	return []*core.Tensor{c.Weights, c.Bias}
}

// SetParameters sets the layer's parameters
func (c *Conv2DLayer) SetParameters(params []*core.Tensor) {
	if len(params) != 2 {
		panic("Conv2DLayer requires exactly 2 parameters (weights, bias)")
	}
	c.Weights = params[0]
	c.Bias = params[1]
}

// MaxPool2DLayer represents a 2D max pooling layer
type MaxPool2DLayer struct {
	PoolSize  [2]int
	Stride    [2]int
	Padding   [2]int
	LastInput *core.Tensor
	LastMask  *core.Tensor
}

// NewMaxPool2DLayer creates a new 2D max pooling layer
func NewMaxPool2DLayer(poolSize, stride, padding [2]int) *MaxPool2DLayer {
	return &MaxPool2DLayer{
		PoolSize: poolSize,
		Stride:   stride,
		Padding:  padding,
	}
}

// Forward performs forward propagation for 2D max pooling
func (m *MaxPool2DLayer) Forward(input *core.Tensor) *core.Tensor {
	m.LastInput = input

	batchSize := input.Shape[0]
	channels := input.Shape[1]
	height := input.Shape[2]
	width := input.Shape[3]

	// Calculate output dimensions
	outHeight := (height+2*m.Padding[0]-m.PoolSize[0])/m.Stride[0] + 1
	outWidth := (width+2*m.Padding[1]-m.PoolSize[1])/m.Stride[1] + 1

	output := core.NewTensor([]int{batchSize, channels, outHeight, outWidth})
	mask := core.NewTensor([]int{batchSize, channels, outHeight, outWidth})

	for b := 0; b < batchSize; b++ {
		for c := 0; c < channels; c++ {
			for h := 0; h < outHeight; h++ {
				for w := 0; w < outWidth; w++ {
					maxVal := math.Inf(-1)
					maxIdx := -1

					// Find maximum in pool region
					for ph := 0; ph < m.PoolSize[0]; ph++ {
						for pw := 0; pw < m.PoolSize[1]; pw++ {
							inH := h*m.Stride[0] + ph - m.Padding[0]
							inW := w*m.Stride[1] + pw - m.Padding[1]

							if inH >= 0 && inH < height && inW >= 0 && inW < width {
								val := input.At(b, c, inH, inW)
								if val > maxVal {
									maxVal = val
									maxIdx = ph*m.PoolSize[1] + pw
								}
							}
						}
					}

					output.Set(maxVal, b, c, h, w)
					mask.Set(float64(maxIdx), b, c, h, w)
				}
			}
		}
	}

	m.LastMask = mask
	return output
}

// Backward performs backward propagation for 2D max pooling
func (m *MaxPool2DLayer) Backward(gradient *core.Tensor) *core.Tensor {
	inputGradient := core.Zeros(m.LastInput.Shape)

	batchSize := gradient.Shape[0]
	channels := gradient.Shape[1]
	outHeight := gradient.Shape[2]
	outWidth := gradient.Shape[3]

	for b := 0; b < batchSize; b++ {
		for c := 0; c < channels; c++ {
			for h := 0; h < outHeight; h++ {
				for w := 0; w < outWidth; w++ {
					maxIdx := int(m.LastMask.At(b, c, h, w))
					ph := maxIdx / m.PoolSize[1]
					pw := maxIdx % m.PoolSize[1]

					inH := h*m.Stride[0] + ph - m.Padding[0]
					inW := w*m.Stride[1] + pw - m.Padding[1]

					if inH >= 0 && inH < m.LastInput.Shape[2] && inW >= 0 && inW < m.LastInput.Shape[3] {
						inputGradient.Set(gradient.At(b, c, h, w), b, c, inH, inW)
					}
				}
			}
		}
	}

	return inputGradient
}

// Parameters returns empty slice (max pooling has no parameters)
func (m *MaxPool2DLayer) Parameters() []*core.Tensor {
	return []*core.Tensor{}
}

// SetParameters does nothing (max pooling has no parameters)
func (m *MaxPool2DLayer) SetParameters(params []*core.Tensor) {
	// No parameters to set
}

// FlattenLayer flattens multi-dimensional input to 1D
type FlattenLayer struct {
	InputShape []int
}

// NewFlattenLayer creates a new flatten layer
func NewFlattenLayer() *FlattenLayer {
	return &FlattenLayer{}
}

// Forward flattens the input
func (f *FlattenLayer) Forward(input *core.Tensor) *core.Tensor {
	f.InputShape = input.Shape

	// Calculate total size
	totalSize := 1
	for _, dim := range input.Shape {
		totalSize *= dim
	}

	// Reshape to [batch, features]
	batchSize := input.Shape[0]
	features := totalSize / batchSize

	return input.Reshape([]int{batchSize, features})
}

// Backward reshapes back to original shape
func (f *FlattenLayer) Backward(gradient *core.Tensor) *core.Tensor {
	return gradient.Reshape(f.InputShape)
}

// Parameters returns empty slice (flatten has no parameters)
func (f *FlattenLayer) Parameters() []*core.Tensor {
	return []*core.Tensor{}
}

// SetParameters does nothing (flatten has no parameters)
func (f *FlattenLayer) SetParameters(params []*core.Tensor) {
	// No parameters to set
}

// DropoutLayer implements dropout regularization
type DropoutLayer struct {
	Rate      float64
	Training  bool
	LastInput *core.Tensor
	Mask      *core.Tensor
}

// NewDropoutLayer creates a new dropout layer
func NewDropoutLayer(rate float64) *DropoutLayer {
	return &DropoutLayer{
		Rate:     rate,
		Training: true,
	}
}

// Forward applies dropout during training
func (d *DropoutLayer) Forward(input *core.Tensor) *core.Tensor {
	d.LastInput = input

	if !d.Training {
		return input
	}

	// Create dropout mask
	mask := core.Random(input.Shape)
	mask = mask.Apply(func(x float64) float64 {
		if x < d.Rate {
			return 0
		}
		return 1.0 / (1.0 - d.Rate)
	})

	d.Mask = mask
	return input.Mul(mask)
}

// Backward applies dropout mask to gradient
func (d *DropoutLayer) Backward(gradient *core.Tensor) *core.Tensor {
	if !d.Training {
		return gradient
	}
	return gradient.Mul(d.Mask)
}

// Parameters returns empty slice (dropout has no parameters)
func (d *DropoutLayer) Parameters() []*core.Tensor {
	return []*core.Tensor{}
}

// SetParameters does nothing (dropout has no parameters)
func (d *DropoutLayer) SetParameters(params []*core.Tensor) {
	// No parameters to set
}

// SetTraining sets the training mode
func (d *DropoutLayer) SetTraining(training bool) {
	d.Training = training
}

// BatchNormLayer implements batch normalization
type BatchNormLayer struct {
	Gamma       *core.Tensor
	Beta        *core.Tensor
	RunningMean *core.Tensor
	RunningVar  *core.Tensor
	Momentum    float64
	Epsilon     float64
	Training    bool
	LastInput   *core.Tensor
	LastMean    *core.Tensor
	LastVar     *core.Tensor
}

// NewBatchNormLayer creates a new batch normalization layer
func NewBatchNormLayer(numFeatures int) *BatchNormLayer {
	return &BatchNormLayer{
		Gamma:       core.Ones([]int{numFeatures}),
		Beta:        core.Zeros([]int{numFeatures}),
		RunningMean: core.Zeros([]int{numFeatures}),
		RunningVar:  core.Ones([]int{numFeatures}),
		Momentum:    0.1,
		Epsilon:     1e-5,
		Training:    true,
	}
}

// Forward performs batch normalization
func (b *BatchNormLayer) Forward(input *core.Tensor) *core.Tensor {
	b.LastInput = input

	if input.Dims() != 2 {
		panic("BatchNormLayer requires 2D input")
	}

	batchSize := input.Shape[0]
	numFeatures := input.Shape[1]

	var mean, var_ *core.Tensor

	if b.Training {
		// Calculate batch statistics
		mean = core.Zeros([]int{numFeatures})
		for i := 0; i < numFeatures; i++ {
			sum := 0.0
			for j := 0; j < batchSize; j++ {
				sum += input.At(j, i)
			}
			mean.Set(sum/float64(batchSize), i)
		}

		var_ = core.Zeros([]int{numFeatures})
		for i := 0; i < numFeatures; i++ {
			sum := 0.0
			for j := 0; j < batchSize; j++ {
				diff := input.At(j, i) - mean.At(i)
				sum += diff * diff
			}
			var_.Set(sum/float64(batchSize), i)
		}

		// Update running statistics
		for i := 0; i < numFeatures; i++ {
			b.RunningMean.Set(b.Momentum*b.RunningMean.At(i)+(1-b.Momentum)*mean.At(i), i)
			b.RunningVar.Set(b.Momentum*b.RunningVar.At(i)+(1-b.Momentum)*var_.At(i), i)
		}

		b.LastMean = mean
		b.LastVar = var_
	} else {
		mean = b.RunningMean
		var_ = b.RunningVar
	}

	// Normalize
	normalized := core.NewTensor(input.Shape)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < numFeatures; j++ {
			norm := (input.At(i, j) - mean.At(j)) / math.Sqrt(var_.At(j)+b.Epsilon)
			normalized.Set(norm, i, j)
		}
	}

	// Scale and shift
	output := core.NewTensor(input.Shape)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < numFeatures; j++ {
			output.Set(normalized.At(i, j)*b.Gamma.At(j)+b.Beta.At(j), i, j)
		}
	}

	return output
}

// Backward performs backward propagation for batch normalization
func (b *BatchNormLayer) Backward(gradient *core.Tensor) *core.Tensor {
	// Simplified backward pass
	return gradient
}

// Parameters returns the layer's parameters
func (b *BatchNormLayer) Parameters() []*core.Tensor {
	return []*core.Tensor{b.Gamma, b.Beta}
}

// SetParameters sets the layer's parameters
func (b *BatchNormLayer) SetParameters(params []*core.Tensor) {
	if len(params) != 2 {
		panic("BatchNormLayer requires exactly 2 parameters (gamma, beta)")
	}
	b.Gamma = params[0]
	b.Beta = params[1]
}

// SetTraining sets the training mode
func (b *BatchNormLayer) SetTraining(training bool) {
	b.Training = training
}
