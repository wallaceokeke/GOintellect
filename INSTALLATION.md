# gointellect Installation Guide

## 🚀 Installing gointellect

gointellect is designed to be easily installable by Go developers worldwide. Here's how to get started:

## 📦 Installation Methods

### 1. Standard Go Module Installation

```bash
# Install the latest stable version
go get github.com/wallaceokeke/GOintellect

# Install a specific version
go get github.com/wallaceokeke/GOintellect@v1.0.0

# Install the latest development version
go get github.com/wallaceokeke/GOintellect@main

# Update to the latest version
go get -u github.com/wallaceokeke/GOintellect
```

### 2. Using go mod init (for new projects)

```bash
# Create a new project
mkdir my-ml-project
cd my-ml-project

# Initialize Go module
go mod init my-ml-project

# Add gointellect dependency
go get github.com/gointellect/gointellect

# Create your first ML program
cat > main.go << 'EOF'
package main

import (
    "fmt"
    "github.com/wallaceokeke/GOintellect/pkg/core"
    "github.com/wallaceokeke/GOintellect/pkg/nn"
    "github.com/wallaceokeke/GOintellect/pkg/train"
)

func main() {
    // Create a simple neural network
    model := nn.NewNeuralNetwork(
        nn.NewDenseLayer(4, 8, nn.NewReLU()),
        nn.NewDenseLayer(8, 2, nn.NewSoftmax()),
    )
    
    fmt.Println("Neural network created successfully!")
    fmt.Printf("Model has %d parameters\n", len(model.Parameters()))
}
EOF

# Run your program
go run main.go
```

### 3. Docker Installation

```bash
# Pull the official gointellect Docker image
docker pull GOintellect/GOintellect:latest

# Run the comprehensive demo
docker run --rm GOintellect/GOintellect:latest go run examples/comprehensive_demo.go

# Run the steroids-level demo
docker run --rm GOintellect/GOintellect:latest go run examples/steroids_demo.go
```

### 4. Building from Source

```bash
# Clone the repository
git clone https://github.com/wallaceokeke/GOintellect.git
cd GOintellect

# Build the CLI tool
go build -o GOintellect ./cmd/GOintellect

# Run examples
go run examples/demo_train_predict.go
go run examples/comprehensive_demo.go
go run examples/steroids_demo.go
```

## 🎯 Quick Start Examples

### Example 1: Basic Tensor Operations

```go
package main

import (
    "fmt"
    "github.com/wallaceokeke/GGOintellect/pkg/core"
)

func main() {
    // Create tensors
    a := core.NewTensor2D(2, 3)
    b := core.NewTensor2D(2, 3)
    
    // Fill with data
    for i := 0; i < 2; i++ {
        for j := 0; j < 3; j++ {
            a.Set(float64(i*3+j+1), i, j)
            b.Set(float64((i*3+j+1)*2), i, j)
        }
    }
    
    // Element-wise operations
    c := a.Add(b)
    d := a.Mul(b)
    
    fmt.Printf("A + B:\n%v\n", c)
    fmt.Printf("A * B:\n%v\n", d)
}
```

### Example 2: Neural Network Training

```go
package main

import (
    "fmt"
    "math/rand"
    "time"
    
    "github.com/wallaceokeke/GOintellect/pkg/core"
    "github.com/wallaceokeke/GOintellect/pkg/nn"
    "github.com/wallaceokeke/GOintellect/pkg/train"
)

func main() {
    rand.Seed(time.Now().UnixNano())
    
    // Create synthetic data
    X := core.NewTensor2D(100, 4)
    y := core.NewTensor2D(100, 2)
    
    for i := 0; i < 100; i++ {
        for j := 0; j < 4; j++ {
            X.Set(rand.Float64()*10, i, j)
        }
        
        // Create classification targets
        if i < 50 {
            y.Set(1, i, 0)
            y.Set(0, i, 1)
        } else {
            y.Set(0, i, 0)
            y.Set(1, i, 1)
        }
    }
    
    // Create neural network
    model := nn.NewNeuralNetwork(
        nn.NewDenseLayer(4, 8, nn.NewReLU()),
        nn.NewDenseLayer(8, 2, nn.NewSoftmax()),
    )
    
    // Create trainer
    loss := nn.NewCrossEntropy()
    optimizer := nn.NewAdam(0.001)
    trainer := train.NewTrainer(model, loss, optimizer)
    trainer.AddMetric(train.NewAccuracy())
    
    // Training configuration
    config := train.TrainConfig{
        Epochs:    50,
        BatchSize: 16,
        Shuffle:   true,
    }
    
    // Train the model
    result := trainer.Train(X, y, config)
    fmt.Printf("Training completed in %v\n", result.TrainingTime)
    fmt.Printf("Final accuracy: %.4f\n", result.History["accuracy"][len(result.History["accuracy"])-1])
}
```

### Example 3: Data Pipeline

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/wallaceokeke/GOintellect/pkg/core"
    "github.com/wallaceokeke/GOintellect/pkg/data"
)

func main() {
    // Create synthetic dataset
    features := core.NewTensor2D(50, 4)
    targets := core.NewTensor2D(50, 1)
    
    for i := 0; i < 50; i++ {
        for j := 0; j < 4; j++ {
            features.Set(float64(i*4+j), i, j)
        }
        targets.Set(float64(i*2+1), i, 0)
    }
    
    dataset := data.NewDataset(features, targets)
    
    // Apply standardization
    scaler := data.NewStandardScaler()
    scaledDataset, err := scaler.FitTransform(dataset)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Original dataset shape: %v\n", dataset.Features.Shape)
    fmt.Printf("Scaled dataset shape: %v\n", scaledDataset.Features.Shape)
    fmt.Printf("Scaled features mean: %.4f\n", scaledDataset.Features.Mean())
}
```

## 🔧 Requirements

- **Go Version**: Go 1.22 or later
- **Operating System**: Windows, macOS, Linux
- **Architecture**: x86_64, ARM64
- **Dependencies**: None (pure Go implementation)

## 🚀 Verification

After installation, verify that gointellect is working correctly:

```bash
# Check if the module is available
go list -m github.com/wallaceokeke/GOintellect

# Run the basic demo
go run -m github.com/wallaceokeke/GOintellect/examples/demo_train_predict.go

# Run the comprehensive demo
go run -m github.com/wallaceokeke/GOintellect/examples/comprehensive_demo.go
```

## 🐛 Troubleshooting

### Common Issues

1. **Module not found**
   ```bash
   # Ensure Go modules are enabled
   export GO111MODULE=on
   go get github.com/wallaceokeke/GOintellect
   ```

2. **Version conflicts**
   ```bash
   # Clean module cache
   go clean -modcache
   go get github.com/wallaceokeke/GOintellect
   ```

3. **Build errors**
   ```bash
   # Update Go to latest version
   go version
   # If needed, download latest Go from https://golang.org/dl/
   ```

### Getting Help

- **GitHub Issues**: [Report bugs and request features](https://github.com/wallaceokeke/GOintellect/issues)
- **Documentation**: Check the [README.md](https://github.com/wallaceokeke/GOintellect/blob/main/README.md)
- **Examples**: Run the provided examples in the `examples/` directory

## 🎯 Next Steps

1. **Explore Examples**: Run the provided examples to understand the library
2. **Read Documentation**: Check the README and package documentation
3. **Build Your First Model**: Start with a simple classification or regression task
4. **Join the Community**: Star the repository and contribute to the project

---

**Welcome to gointellect!** 🚀 Start building amazing ML applications in Go today!
