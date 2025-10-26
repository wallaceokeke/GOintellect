package core

import (
	"fmt"
	"math"
	"math/rand"
)

// Tensor represents an N-dimensional array with operations similar to NumPy
type Tensor struct {
	Shape []int
	Data  []float64
	Stride []int
}

// NewTensor creates a new tensor with the given shape
func NewTensor(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	
	stride := make([]int, len(shape))
	stride[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		stride[i] = stride[i+1] * shape[i+1]
	}
	
	return &Tensor{
		Shape:  shape,
		Data:   make([]float64, size),
		Stride: stride,
	}
}

// NewTensorFromData creates a tensor from existing data
func NewTensorFromData(data []float64, shape []int) *Tensor {
	t := NewTensor(shape)
	copy(t.Data, data)
	return t
}

// NewTensor2D creates a 2D tensor (matrix)
func NewTensor2D(rows, cols int) *Tensor {
	return NewTensor([]int{rows, cols})
}

// NewTensor1D creates a 1D tensor (vector)
func NewTensor1D(size int) *Tensor {
	return NewTensor([]int{size})
}

// At returns the value at the given indices
func (t *Tensor) At(indices ...int) float64 {
	idx := 0
	for i, index := range indices {
		idx += index * t.Stride[i]
	}
	return t.Data[idx]
}

// Set sets the value at the given indices
func (t *Tensor) Set(value float64, indices ...int) {
	idx := 0
	for i, index := range indices {
		idx += index * t.Stride[i]
	}
	t.Data[idx] = value
}

// Size returns the total number of elements
func (t *Tensor) Size() int {
	size := 1
	for _, dim := range t.Shape {
		size *= dim
	}
	return size
}

// Dims returns the number of dimensions
func (t *Tensor) Dims() int {
	return len(t.Shape)
}

// Reshape returns a new tensor with the given shape
func (t *Tensor) Reshape(newShape []int) *Tensor {
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}
	
	if newSize != t.Size() {
		panic(fmt.Sprintf("cannot reshape tensor of size %d to shape %v", t.Size(), newShape))
	}
	
	newTensor := &Tensor{
		Shape:  newShape,
		Data:   make([]float64, len(t.Data)),
		Stride: make([]int, len(newShape)),
	}
	copy(newTensor.Data, t.Data)
	
	// Calculate new strides
	newTensor.Stride[len(newShape)-1] = 1
	for i := len(newShape) - 2; i >= 0; i-- {
		newTensor.Stride[i] = newTensor.Stride[i+1] * newShape[i+1]
	}
	
	return newTensor
}

// Add performs element-wise addition
func (t *Tensor) Add(other *Tensor) *Tensor {
	if !t.ShapeEqual(other) {
		panic("tensors must have the same shape for addition")
	}
	
	result := NewTensor(t.Shape)
	for i := range t.Data {
		result.Data[i] = t.Data[i] + other.Data[i]
	}
	return result
}

// Mul performs element-wise multiplication
func (t *Tensor) Mul(other *Tensor) *Tensor {
	if !t.ShapeEqual(other) {
		panic("tensors must have the same shape for multiplication")
	}
	
	result := NewTensor(t.Shape)
	for i := range t.Data {
		result.Data[i] = t.Data[i] * other.Data[i]
	}
	return result
}

// Scale multiplies all elements by a scalar
func (t *Tensor) Scale(scalar float64) *Tensor {
	result := NewTensor(t.Shape)
	for i := range t.Data {
		result.Data[i] = t.Data[i] * scalar
	}
	return result
}

// Dot performs matrix multiplication
func (t *Tensor) Dot(other *Tensor) *Tensor {
	if t.Dims() != 2 || other.Dims() != 2 {
		panic("dot product requires 2D tensors")
	}
	
	if t.Shape[1] != other.Shape[0] {
		panic(fmt.Sprintf("incompatible shapes for dot product: %v and %v", t.Shape, other.Shape))
	}
	
	result := NewTensor2D(t.Shape[0], other.Shape[1])
	
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < other.Shape[1]; j++ {
			sum := 0.0
			for k := 0; k < t.Shape[1]; k++ {
				sum += t.At(i, k) * other.At(k, j)
			}
			result.Set(sum, i, j)
		}
	}
	
	return result
}

// Transpose returns the transpose of a 2D tensor
func (t *Tensor) Transpose() *Tensor {
	if t.Dims() != 2 {
		panic("transpose requires 2D tensor")
	}
	
	result := NewTensor2D(t.Shape[1], t.Shape[0])
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			result.Set(t.At(i, j), j, i)
		}
	}
	return result
}

// Sum returns the sum of all elements
func (t *Tensor) Sum() float64 {
	sum := 0.0
	for _, val := range t.Data {
		sum += val
	}
	return sum
}

// Mean returns the mean of all elements
func (t *Tensor) Mean() float64 {
	return t.Sum() / float64(t.Size())
}

// ShapeEqual checks if two tensors have the same shape
func (t *Tensor) ShapeEqual(other *Tensor) bool {
	if len(t.Shape) != len(other.Shape) {
		return false
	}
	for i, dim := range t.Shape {
		if dim != other.Shape[i] {
			return false
		}
	}
	return true
}

// String returns a string representation of the tensor
func (t *Tensor) String() string {
	if t.Dims() == 1 {
		return fmt.Sprintf("Tensor1D%v: %v", t.Shape, t.Data)
	} else if t.Dims() == 2 {
		result := fmt.Sprintf("Tensor2D%v:\n", t.Shape)
		for i := 0; i < t.Shape[0]; i++ {
			row := make([]float64, t.Shape[1])
			for j := 0; j < t.Shape[1]; j++ {
				row[j] = t.At(i, j)
			}
			result += fmt.Sprintf("  %v\n", row)
		}
		return result
	}
	return fmt.Sprintf("Tensor%v: %v", t.Shape, t.Data)
}

// Apply applies a function to each element
func (t *Tensor) Apply(fn func(float64) float64) *Tensor {
	result := NewTensor(t.Shape)
	for i, val := range t.Data {
		result.Data[i] = fn(val)
	}
	return result
}

// Zeros creates a tensor filled with zeros
func Zeros(shape []int) *Tensor {
	return NewTensor(shape)
}

// Ones creates a tensor filled with ones
func Ones(shape []int) *Tensor {
	t := NewTensor(shape)
	for i := range t.Data {
		t.Data[i] = 1.0
	}
	return t
}

// Random creates a tensor with random values between 0 and 1
func Random(shape []int) *Tensor {
	t := NewTensor(shape)
	for i := range t.Data {
		t.Data[i] = rand.Float64()
	}
	return t
}

// RandomNormal creates a tensor with random values from normal distribution
func RandomNormal(shape []int, mean, std float64) *Tensor {
	t := NewTensor(shape)
	for i := range t.Data {
		// Simple Box-Muller transform for normal distribution
		u1 := rand.Float64()
		u2 := rand.Float64()
		z0 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		t.Data[i] = mean + std*z0
	}
	return t
}
