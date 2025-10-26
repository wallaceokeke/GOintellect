package nn

import (
	"math"

	"github.com/gointellect/gointellect/pkg/core"
)

// RNNLayer represents a basic RNN layer
type RNNLayer struct {
	HiddenSize  int
	InputSize   int
	Wx          *core.Tensor // Input weights
	Wh          *core.Tensor // Hidden weights
	Bias        *core.Tensor
	Activation  ActivationFunction
	LastHidden  *core.Tensor
	LastInput   *core.Tensor
}

// NewRNNLayer creates a new RNN layer
func NewRNNLayer(inputSize, hiddenSize int, activation ActivationFunction) *RNNLayer {
	// Initialize weights with Xavier initialization
	wxLimit := math.Sqrt(6.0 / float64(inputSize + hiddenSize))
	whLimit := math.Sqrt(6.0 / float64(hiddenSize + hiddenSize))
	
	wx := core.Random([]int{inputSize, hiddenSize}).Scale(2 * wxLimit).Add(core.Random([]int{inputSize, hiddenSize}).Scale(-wxLimit))
	wh := core.Random([]int{hiddenSize, hiddenSize}).Scale(2 * whLimit).Add(core.Random([]int{hiddenSize, hiddenSize}).Scale(-whLimit))
	bias := core.Zeros([]int{hiddenSize})
	
	return &RNNLayer{
		HiddenSize: hiddenSize,
		InputSize:  inputSize,
		Wx:         wx,
		Wh:         wh,
		Bias:       bias,
		Activation: activation,
	}
}

// Forward performs forward propagation for RNN
func (r *RNNLayer) Forward(input *core.Tensor) *core.Tensor {
	r.LastInput = input
	
	// Input shape: [batch, sequence, features]
	batchSize := input.Shape[0]
	seqLen := input.Shape[1]
	
	// Initialize hidden states
	hidden := core.Zeros([]int{batchSize, seqLen, r.HiddenSize})
	
	// Process each time step
	for t := 0; t < seqLen; t++ {
		// Current input: [batch, features]
		xt := core.NewTensor2D(batchSize, r.InputSize)
		for b := 0; b < batchSize; b++ {
			for f := 0; f < r.InputSize; f++ {
				xt.Set(input.At(b, t, f), b, f)
			}
		}
		
		// Previous hidden state
		var htPrev *core.Tensor
		if t == 0 {
			htPrev = core.Zeros([]int{batchSize, r.HiddenSize})
		} else {
			htPrev = core.NewTensor2D(batchSize, r.HiddenSize)
			for b := 0; b < batchSize; b++ {
				for h := 0; h < r.HiddenSize; h++ {
					htPrev.Set(hidden.At(b, t-1, h), b, h)
				}
			}
		}
		
		// Compute new hidden state: ht = activation(Wx * xt + Wh * ht-1 + b)
		wxOut := xt.Dot(r.Wx)
		whOut := htPrev.Dot(r.Wh)
		
		// Add bias
		for b := 0; b < batchSize; b++ {
			for h := 0; h < r.HiddenSize; h++ {
				sum := wxOut.At(b, h) + whOut.At(b, h) + r.Bias.At(h)
				hidden.Set(sum, b, t, h)
			}
		}
		
		// Apply activation
		if r.Activation != nil {
			ht := core.NewTensor2D(batchSize, r.HiddenSize)
			for b := 0; b < batchSize; b++ {
				for h := 0; h < r.HiddenSize; h++ {
					ht.Set(hidden.At(b, t, h), b, h)
				}
			}
			ht = r.Activation.Forward(ht)
			for b := 0; b < batchSize; b++ {
				for h := 0; h < r.HiddenSize; h++ {
					hidden.Set(ht.At(b, h), b, t, h)
				}
			}
		}
	}
	
	r.LastHidden = hidden
	return hidden
}

// Backward performs backward propagation for RNN
func (r *RNNLayer) Backward(gradient *core.Tensor) *core.Tensor {
	// Simplified backward pass
	return gradient
}

// Parameters returns the layer's parameters
func (r *RNNLayer) Parameters() []*core.Tensor {
	return []*core.Tensor{r.Wx, r.Wh, r.Bias}
}

// SetParameters sets the layer's parameters
func (r *RNNLayer) SetParameters(params []*core.Tensor) {
	if len(params) != 3 {
		panic("RNNLayer requires exactly 3 parameters (Wx, Wh, bias)")
	}
	r.Wx = params[0]
	r.Wh = params[1]
	r.Bias = params[2]
}

// LSTMLayer represents an LSTM layer
type LSTMLayer struct {
	HiddenSize int
	InputSize  int
	Wf         *core.Tensor // Forget gate weights
	Wi         *core.Tensor // Input gate weights
	Wc         *core.Tensor // Candidate gate weights
	Wo         *core.Tensor // Output gate weights
	Bias       *core.Tensor
	LastHidden *core.Tensor
	LastCell   *core.Tensor
	LastInput  *core.Tensor
}

// NewLSTMLayer creates a new LSTM layer
func NewLSTMLayer(inputSize, hiddenSize int) *LSTMLayer {
	// Initialize weights
	wLimit := math.Sqrt(6.0 / float64(inputSize + hiddenSize))
	
	wf := core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(2 * wLimit).Add(core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(-wLimit))
	wi := core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(2 * wLimit).Add(core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(-wLimit))
	wc := core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(2 * wLimit).Add(core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(-wLimit))
	wo := core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(2 * wLimit).Add(core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(-wLimit))
	bias := core.Zeros([]int{4 * hiddenSize})
	
	return &LSTMLayer{
		HiddenSize: hiddenSize,
		InputSize:  inputSize,
		Wf:         wf,
		Wi:         wi,
		Wc:         wc,
		Wo:         wo,
		Bias:       bias,
	}
}

// Forward performs forward propagation for LSTM
func (l *LSTMLayer) Forward(input *core.Tensor) *core.Tensor {
	l.LastInput = input
	
	// Input shape: [batch, sequence, features]
	batchSize := input.Shape[0]
	seqLen := input.Shape[1]
	
	// Initialize hidden and cell states
	hidden := core.Zeros([]int{batchSize, seqLen, l.HiddenSize})
	cell := core.Zeros([]int{batchSize, seqLen, l.HiddenSize})
	
	// Process each time step
	for t := 0; t < seqLen; t++ {
		// Current input: [batch, features]
		xt := core.NewTensor2D(batchSize, l.InputSize)
		for b := 0; b < batchSize; b++ {
			for f := 0; f < l.InputSize; f++ {
				xt.Set(input.At(b, t, f), b, f)
			}
		}
		
		// Previous hidden state
		var htPrev *core.Tensor
		if t == 0 {
			htPrev = core.Zeros([]int{batchSize, l.HiddenSize})
		} else {
			htPrev = core.NewTensor2D(batchSize, l.HiddenSize)
			for b := 0; b < batchSize; b++ {
				for h := 0; h < l.HiddenSize; h++ {
					htPrev.Set(hidden.At(b, t-1, h), b, h)
				}
			}
		}
		
		// Previous cell state
		var ctPrev *core.Tensor
		if t == 0 {
			ctPrev = core.Zeros([]int{batchSize, l.HiddenSize})
		} else {
			ctPrev = core.NewTensor2D(batchSize, l.HiddenSize)
			for b := 0; b < batchSize; b++ {
				for h := 0; h < l.HiddenSize; h++ {
					ctPrev.Set(cell.At(b, t-1, h), b, h)
				}
			}
		}
		
		// Concatenate input and previous hidden state
		concat := core.NewTensor2D(batchSize, l.InputSize+l.HiddenSize)
		for b := 0; b < batchSize; b++ {
			for i := 0; i < l.InputSize; i++ {
				concat.Set(xt.At(b, i), b, i)
			}
			for i := 0; i < l.HiddenSize; i++ {
				concat.Set(htPrev.At(b, i), b, l.InputSize+i)
			}
		}
		
		// Compute gates
		// Forget gate: ft = sigmoid(Wf * [xt, ht-1] + bf)
		ft := concat.Dot(l.Wf)
		for b := 0; b < batchSize; b++ {
			for h := 0; h < l.HiddenSize; h++ {
				ft.Set(1.0/(1.0+math.Exp(-ft.At(b, h))), b, h)
			}
		}
		
		// Input gate: it = sigmoid(Wi * [xt, ht-1] + bi)
		it := concat.Dot(l.Wi)
		for b := 0; b < batchSize; b++ {
			for h := 0; h < l.HiddenSize; h++ {
				it.Set(1.0/(1.0+math.Exp(-it.At(b, h))), b, h)
			}
		}
		
		// Candidate values: Ct = tanh(Wc * [xt, ht-1] + bc)
		ct := concat.Dot(l.Wc)
		for b := 0; b < batchSize; b++ {
			for h := 0; h < l.HiddenSize; h++ {
				ct.Set(math.Tanh(ct.At(b, h)), b, h)
			}
		}
		
		// Output gate: ot = sigmoid(Wo * [xt, ht-1] + bo)
		ot := concat.Dot(l.Wo)
		for b := 0; b < batchSize; b++ {
			for h := 0; h < l.HiddenSize; h++ {
				ot.Set(1.0/(1.0+math.Exp(-ot.At(b, h))), b, h)
			}
		}
		
		// Update cell state: Ct = ft * Ct-1 + it * Ct
		for b := 0; b < batchSize; b++ {
			for h := 0; h < l.HiddenSize; h++ {
				newCell := ft.At(b, h)*ctPrev.At(b, h) + it.At(b, h)*ct.At(b, h)
				cell.Set(newCell, b, t, h)
			}
		}
		
		// Compute hidden state: ht = ot * tanh(Ct)
		for b := 0; b < batchSize; b++ {
			for h := 0; h < l.HiddenSize; h++ {
				newHidden := ot.At(b, h) * math.Tanh(cell.At(b, t, h))
				hidden.Set(newHidden, b, t, h)
			}
		}
	}
	
	l.LastHidden = hidden
	l.LastCell = cell
	return hidden
}

// Backward performs backward propagation for LSTM
func (l *LSTMLayer) Backward(gradient *core.Tensor) *core.Tensor {
	// Simplified backward pass
	return gradient
}

// Parameters returns the layer's parameters
func (l *LSTMLayer) Parameters() []*core.Tensor {
	return []*core.Tensor{l.Wf, l.Wi, l.Wc, l.Wo, l.Bias}
}

// SetParameters sets the layer's parameters
func (l *LSTMLayer) SetParameters(params []*core.Tensor) {
	if len(params) != 5 {
		panic("LSTMLayer requires exactly 5 parameters (Wf, Wi, Wc, Wo, bias)")
	}
	l.Wf = params[0]
	l.Wi = params[1]
	l.Wc = params[2]
	l.Wo = params[3]
	l.Bias = params[4]
}

// GRULayer represents a GRU layer
type GRULayer struct {
	HiddenSize int
	InputSize  int
	Wz         *core.Tensor // Update gate weights
	Wr         *core.Tensor // Reset gate weights
	Wh         *core.Tensor // Hidden weights
	Bias       *core.Tensor
	LastHidden *core.Tensor
	LastInput  *core.Tensor
}

// NewGRULayer creates a new GRU layer
func NewGRULayer(inputSize, hiddenSize int) *GRULayer {
	// Initialize weights
	wLimit := math.Sqrt(6.0 / float64(inputSize + hiddenSize))
	
	wz := core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(2 * wLimit).Add(core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(-wLimit))
	wr := core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(2 * wLimit).Add(core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(-wLimit))
	wh := core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(2 * wLimit).Add(core.Random([]int{inputSize + hiddenSize, hiddenSize}).Scale(-wLimit))
	bias := core.Zeros([]int{3 * hiddenSize})
	
	return &GRULayer{
		HiddenSize: hiddenSize,
		InputSize:  inputSize,
		Wz:         wz,
		Wr:         wr,
		Wh:         wh,
		Bias:       bias,
	}
}

// Forward performs forward propagation for GRU
func (g *GRULayer) Forward(input *core.Tensor) *core.Tensor {
	g.LastInput = input
	
	// Input shape: [batch, sequence, features]
	batchSize := input.Shape[0]
	seqLen := input.Shape[1]
	
	// Initialize hidden states
	hidden := core.Zeros([]int{batchSize, seqLen, g.HiddenSize})
	
	// Process each time step
	for t := 0; t < seqLen; t++ {
		// Current input: [batch, features]
		xt := core.NewTensor2D(batchSize, g.InputSize)
		for b := 0; b < batchSize; b++ {
			for f := 0; f < g.InputSize; f++ {
				xt.Set(input.At(b, t, f), b, f)
			}
		}
		
		// Previous hidden state
		var htPrev *core.Tensor
		if t == 0 {
			htPrev = core.Zeros([]int{batchSize, g.HiddenSize})
		} else {
			htPrev = core.NewTensor2D(batchSize, g.HiddenSize)
			for b := 0; b < batchSize; b++ {
				for h := 0; h < g.HiddenSize; h++ {
					htPrev.Set(hidden.At(b, t-1, h), b, h)
				}
			}
		}
		
		// Concatenate input and previous hidden state
		concat := core.NewTensor2D(batchSize, g.InputSize+g.HiddenSize)
		for b := 0; b < batchSize; b++ {
			for i := 0; i < g.InputSize; i++ {
				concat.Set(xt.At(b, i), b, i)
			}
			for i := 0; i < g.HiddenSize; i++ {
				concat.Set(htPrev.At(b, i), b, g.InputSize+i)
			}
		}
		
		// Update gate: zt = sigmoid(Wz * [xt, ht-1] + bz)
		zt := concat.Dot(g.Wz)
		for b := 0; b < batchSize; b++ {
			for h := 0; h < g.HiddenSize; h++ {
				zt.Set(1.0/(1.0+math.Exp(-zt.At(b, h))), b, h)
			}
		}
		
		// Reset gate: rt = sigmoid(Wr * [xt, ht-1] + br)
		rt := concat.Dot(g.Wr)
		for b := 0; b < batchSize; b++ {
			for h := 0; h < g.HiddenSize; h++ {
				rt.Set(1.0/(1.0+math.Exp(-rt.At(b, h))), b, h)
			}
		}
		
		// Candidate hidden state: ht = tanh(Wh * [xt, rt * ht-1] + bh)
		rtHtPrev := core.NewTensor2D(batchSize, g.HiddenSize)
		for b := 0; b < batchSize; b++ {
			for h := 0; h < g.HiddenSize; h++ {
				rtHtPrev.Set(rt.At(b, h)*htPrev.At(b, h), b, h)
			}
		}
		
		concat2 := core.NewTensor2D(batchSize, g.InputSize+g.HiddenSize)
		for b := 0; b < batchSize; b++ {
			for i := 0; i < g.InputSize; i++ {
				concat2.Set(xt.At(b, i), b, i)
			}
			for i := 0; i < g.HiddenSize; i++ {
				concat2.Set(rtHtPrev.At(b, i), b, g.InputSize+i)
			}
		}
		
		ht := concat2.Dot(g.Wh)
		for b := 0; b < batchSize; b++ {
			for h := 0; h < g.HiddenSize; h++ {
				ht.Set(math.Tanh(ht.At(b, h)), b, h)
			}
		}
		
		// Final hidden state: ht = (1 - zt) * ht-1 + zt * ht
		for b := 0; b < batchSize; b++ {
			for h := 0; h < g.HiddenSize; h++ {
				newHidden := (1-zt.At(b, h))*htPrev.At(b, h) + zt.At(b, h)*ht.At(b, h)
				hidden.Set(newHidden, b, t, h)
			}
		}
	}
	
	g.LastHidden = hidden
	return hidden
}

// Backward performs backward propagation for GRU
func (g *GRULayer) Backward(gradient *core.Tensor) *core.Tensor {
	// Simplified backward pass
	return gradient
}

// Parameters returns the layer's parameters
func (g *GRULayer) Parameters() []*core.Tensor {
	return []*core.Tensor{g.Wz, g.Wr, g.Wh, g.Bias}
}

// SetParameters sets the layer's parameters
func (g *GRULayer) SetParameters(params []*core.Tensor) {
	if len(params) != 4 {
		panic("GRULayer requires exactly 4 parameters (Wz, Wr, Wh, bias)")
	}
	g.Wz = params[0]
	g.Wr = params[1]
	g.Wh = params[2]
	g.Bias = params[3]
}
