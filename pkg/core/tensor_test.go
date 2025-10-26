package core

import (
	"math"
	"testing"
)

func TestNewTensor(t *testing.T) {
	tests := []struct {
		name   string
		shape  []int
		expect int
	}{
		{
			name:   "2D tensor",
			shape:  []int{3, 4},
			expect: 12,
		},
		{
			name:   "3D tensor",
			shape:  []int{2, 3, 4},
			expect: 24,
		},
		{
			name:   "1D tensor",
			shape:  []int{10},
			expect: 10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := NewTensor(tt.shape)
			if tensor.Size() != tt.expect {
				t.Errorf("Expected size %d, got %d", tt.expect, tensor.Size())
			}
			if len(tensor.Shape) != len(tt.shape) {
				t.Errorf("Expected shape length %d, got %d", len(tt.shape), len(tensor.Shape))
			}
		})
	}
}

func TestTensorAtSet(t *testing.T) {
	tensor := NewTensor2D(2, 3)
	
	// Test Set and At
	tensor.Set(5.0, 0, 1)
	if tensor.At(0, 1) != 5.0 {
		t.Errorf("Expected 5.0, got %f", tensor.At(0, 1))
	}
	
	// Test bounds checking
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for out of bounds access")
		}
	}()
	tensor.At(10, 10) // This should panic
}

func TestTensorAdd(t *testing.T) {
	a := NewTensor2D(2, 2)
	b := NewTensor2D(2, 2)
	
	// Fill with test data
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			a.Set(float64(i*2+j+1), i, j)
			b.Set(float64((i*2+j+1)*2), i, j)
		}
	}
	
	c := a.Add(b)
	
	// Check results
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			expected := a.At(i, j) + b.At(i, j)
			if c.At(i, j) != expected {
				t.Errorf("At (%d,%d): expected %f, got %f", i, j, expected, c.At(i, j))
			}
		}
	}
}

func TestTensorMul(t *testing.T) {
	a := NewTensor2D(2, 2)
	b := NewTensor2D(2, 2)
	
	// Fill with test data
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			a.Set(2.0, i, j)
			b.Set(3.0, i, j)
		}
	}
	
	c := a.Mul(b)
	
	// Check results
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			expected := 6.0
			if c.At(i, j) != expected {
				t.Errorf("At (%d,%d): expected %f, got %f", i, j, expected, c.At(i, j))
			}
		}
	}
}

func TestTensorDot(t *testing.T) {
	a := NewTensor2D(2, 3)
	b := NewTensor2D(3, 2)
	
	// Fill with test data
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			a.Set(float64(i*3+j+1), i, j)
		}
	}
	
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			b.Set(float64(i*2+j+1), i, j)
		}
	}
	
	c := a.Dot(b)
	
	// Check output shape
	if c.Shape[0] != 2 || c.Shape[1] != 2 {
		t.Errorf("Expected shape [2,2], got %v", c.Shape)
	}
	
	// Check first element: (1*1 + 2*2 + 3*3) = 14
	expected := 1.0*1.0 + 2.0*2.0 + 3.0*3.0
	if math.Abs(c.At(0, 0)-expected) > 1e-10 {
		t.Errorf("Expected %f, got %f", expected, c.At(0, 0))
	}
}

func TestTensorTranspose(t *testing.T) {
	a := NewTensor2D(2, 3)
	
	// Fill with test data
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			a.Set(float64(i*3+j+1), i, j)
		}
	}
	
	b := a.Transpose()
	
	// Check shape
	if b.Shape[0] != 3 || b.Shape[1] != 2 {
		t.Errorf("Expected shape [3,2], got %v", b.Shape)
	}
	
	// Check values
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			if a.At(i, j) != b.At(j, i) {
				t.Errorf("Transpose mismatch at (%d,%d)", i, j)
			}
		}
	}
}

func TestTensorReshape(t *testing.T) {
	a := NewTensor2D(2, 3)
	
	// Fill with test data
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			a.Set(float64(i*3+j+1), i, j)
		}
	}
	
	b := a.Reshape([]int{3, 2})
	
	// Check shape
	if b.Shape[0] != 3 || b.Shape[1] != 2 {
		t.Errorf("Expected shape [3,2], got %v", b.Shape)
	}
	
	// Check that data is preserved
	if a.Data[0] != b.Data[0] {
		t.Error("Data not preserved during reshape")
	}
}

func TestTensorSum(t *testing.T) {
	a := NewTensor2D(2, 2)
	
	// Fill with test data
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			a.Set(1.0, i, j)
		}
	}
	
	sum := a.Sum()
	expected := 4.0
	
	if sum != expected {
		t.Errorf("Expected sum %f, got %f", expected, sum)
	}
}

func TestTensorMean(t *testing.T) {
	a := NewTensor2D(2, 2)
	
	// Fill with test data
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			a.Set(2.0, i, j)
		}
	}
	
	mean := a.Mean()
	expected := 2.0
	
	if mean != expected {
		t.Errorf("Expected mean %f, got %f", expected, mean)
	}
}

func TestTensorApply(t *testing.T) {
	a := NewTensor2D(2, 2)
	
	// Fill with test data
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			a.Set(2.0, i, j)
		}
	}
	
	// Apply square function
	b := a.Apply(func(x float64) float64 {
		return x * x
	})
	
	// Check results
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			expected := 4.0
			if b.At(i, j) != expected {
				t.Errorf("At (%d,%d): expected %f, got %f", i, j, expected, b.At(i, j))
			}
		}
	}
}

func TestTensorScale(t *testing.T) {
	a := NewTensor2D(2, 2)
	
	// Fill with test data
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			a.Set(2.0, i, j)
		}
	}
	
	b := a.Scale(3.0)
	
	// Check results
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			expected := 6.0
			if b.At(i, j) != expected {
				t.Errorf("At (%d,%d): expected %f, got %f", i, j, expected, b.At(i, j))
			}
		}
	}
}

func TestTensorShapeEqual(t *testing.T) {
	a := NewTensor2D(2, 3)
	b := NewTensor2D(2, 3)
	c := NewTensor2D(3, 2)
	
	if !a.ShapeEqual(b) {
		t.Error("Expected shapes to be equal")
	}
	
	if a.ShapeEqual(c) {
		t.Error("Expected shapes to be different")
	}
}

func TestZeros(t *testing.T) {
	zeros := Zeros([]int{2, 3})
	
	for i := 0; i < zeros.Size(); i++ {
		if zeros.Data[i] != 0.0 {
			t.Errorf("Expected 0.0, got %f", zeros.Data[i])
		}
	}
}

func TestOnes(t *testing.T) {
	ones := Ones([]int{2, 3})
	
	for i := 0; i < ones.Size(); i++ {
		if ones.Data[i] != 1.0 {
			t.Errorf("Expected 1.0, got %f", ones.Data[i])
		}
	}
}

func TestRandom(t *testing.T) {
	random := Random([]int{2, 3})
	
	// Check that all values are between 0 and 1
	for i := 0; i < random.Size(); i++ {
		if random.Data[i] < 0.0 || random.Data[i] > 1.0 {
			t.Errorf("Expected value between 0 and 1, got %f", random.Data[i])
		}
	}
}

func TestRandomNormal(t *testing.T) {
	normal := RandomNormal([]int{1000, 1}, 0.0, 1.0)
	
	// Check that values are roughly normal distributed
	mean := normal.Mean()
	if math.Abs(mean) > 0.1 {
		t.Errorf("Expected mean close to 0, got %f", mean)
	}
}

// Benchmark tests
func BenchmarkTensorAdd(b *testing.B) {
	a := NewTensor2D(100, 100)
	b_tensor := NewTensor2D(100, 100)
	
	// Fill with test data
	for i := 0; i < 100; i++ {
		for j := 0; j < 100; j++ {
			a.Set(float64(i*100+j), i, j)
			b_tensor.Set(float64(i*100+j), i, j)
		}
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Add(b_tensor)
	}
}

func BenchmarkTensorDot(b *testing.B) {
	a := NewTensor2D(100, 100)
	b_tensor := NewTensor2D(100, 100)
	
	// Fill with test data
	for i := 0; i < 100; i++ {
		for j := 0; j < 100; j++ {
			a.Set(1.0, i, j)
			b_tensor.Set(1.0, i, j)
		}
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = a.Dot(b_tensor)
	}
}
