# gointellect

**gointellect** — A comprehensive, production-ready machine learning library for Go, designed to fill the gap in the Go ecosystem for ML/AI development. Think TensorFlow/PyTorch for Python, but built specifically for Go developers.

## 🚀 Features

### Core Components
- **`pkg/core`** - Advanced tensor operations with NumPy-like API
- **`pkg/nn`** - Complete neural network framework with layers, activations, optimizers
- **`pkg/train`** - Training framework with callbacks, metrics, and early stopping
- **`pkg/data`** - Comprehensive data pipeline with loaders, transformers, and splitters
- **`pkg/learn`** - Traditional ML algorithms (Linear Regression, Perceptron)

### Key Capabilities
- 🧠 **Neural Networks**: Multi-layer perceptrons, CNNs, RNNs (extensible architecture)
- 📊 **Tensor Operations**: Broadcasting, matrix multiplication, element-wise operations
- 🔄 **Data Pipeline**: CSV loading, preprocessing, normalization, train/test splitting
- 🎯 **Training Framework**: SGD, Adam optimizers, loss functions, metrics, callbacks
- 📈 **Model Evaluation**: Accuracy, precision, recall, MSE, R², cross-validation
- 🚀 **Performance**: Pure Go implementation, no external dependencies
- 🐳 **Production Ready**: Docker support, CLI tools, comprehensive examples

## 📦 Installation

```bash
go get github.com/gointellect/gointellect
```

### Alternative Installation Methods

```bash
# Install specific version
go get github.com/gointellect/gointellect@v1.0.0

# Install latest development version
go get github.com/gointellect/gointellect@main

# Update to latest version
go get -u github.com/gointellect/gointellect
```

## 🎯 Quick Start

### Basic Tensor Operations
```go
package main

import (
    "fmt"
    "github.com/gointellect/gointellect/pkg/core"
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
    
    // Matrix multiplication
    e := core.NewTensor2D(3, 2)
    f := a.Dot(e)
    
    fmt.Printf("A + B:\n%v\n", c)
    fmt.Printf("A * B:\n%v\n", d)
    fmt.Printf("A @ E:\n%v\n", f)
}
```

### Neural Network Training
```go
package main

import (
    "github.com/gointellect/gointellect/pkg/core"
    "github.com/gointellect/gointellect/pkg/nn"
    "github.com/gointellect/gointellect/pkg/train"
)

func main() {
    // Create neural network
    model := nn.NewNeuralNetwork(
        nn.NewDenseLayer(2, 8, nn.NewReLU()),
        nn.NewDenseLayer(8, 4, nn.NewReLU()),
        nn.NewDenseLayer(4, 2, nn.NewSoftmax()),
    )
    
    // Create trainer
    loss := nn.NewCrossEntropy()
    optimizer := nn.NewSGD(0.01)
    trainer := train.NewTrainer(model, loss, optimizer)
    
    // Add metrics
    trainer.AddMetric(train.NewAccuracy())
    
    // Training configuration
    config := train.TrainConfig{
        Epochs:           100,
        BatchSize:        32,
        ValidationSplit: 0.2,
        Shuffle:          true,
        EarlyStopping: train.EarlyStoppingConfig{
            Enabled:  true,
            Patience: 10,
            MinDelta: 0.001,
            Monitor:  "val_loss",
        },
    }
    
    // Train the model
    result := trainer.Train(X, y, config)
    fmt.Printf("Training completed in %v\n", result.TrainingTime)
}
```

### Data Pipeline
```go
package main

import (
    "github.com/gointellect/gointellect/pkg/data"
)

func main() {
    // Load CSV data
    loader := data.NewCSVLoader().
        SetFeatureColumns([]int{0, 1, 2, 3}).
        SetTargetColumns([]int{4})
    
    dataset, err := loader.Load("data.csv")
    if err != nil {
        log.Fatal(err)
    }
    
    // Apply preprocessing
    scaler := data.NewStandardScaler()
    scaledDataset, err := scaler.FitTransform(dataset)
    if err != nil {
        log.Fatal(err)
    }
    
    // Split data
    splitter := data.NewTrainTestSplitter(0.2)
    splits, err := splitter.Split(scaledDataset)
    if err != nil {
        log.Fatal(err)
    }
    
    trainData := splits[0]
    testData := splits[1]
}
```

## 🏗️ Architecture

```
gointellect/
├── pkg/
│   ├── core/           # Core tensor operations
│   ├── nn/             # Neural network framework
│   ├── train/          # Training framework
│   ├── data/           # Data pipeline
│   └── learn/          # Traditional ML algorithms
├── cmd/
│   └── gointellect/    # CLI tool
├── examples/           # Comprehensive examples
└── Dockerfile          # Container support
```

## 🎮 CLI Usage

```bash
# Build the CLI
go build ./cmd/gointellect

# Train a linear regression model
./gointellect train linear "1,2,3,4,5" "3,5,7,9,11"

# Predict with trained model
./gointellect predict linear 2.0 1.0 10

# Train a perceptron
./gointellect train perceptron 2 0.1 100
```

## 🐳 Docker Support

```bash
# Build Docker image
docker build -t gointellect:latest .

# Run examples
docker run --rm gointellect go run examples/comprehensive_demo.go
```

## 📚 Examples

- **`examples/demo_train_predict.go`** - Basic linear regression and perceptron
- **`examples/comprehensive_demo.go`** - Full library demonstration
- **`examples/data/simple_linear.csv`** - Sample dataset

## 🔧 Advanced Features

### Custom Layers
```go
type CustomLayer struct {
    // Implement Layer interface
}

func (c *CustomLayer) Forward(input *core.Tensor) *core.Tensor {
    // Custom forward pass
}

func (c *CustomLayer) Backward(gradient *core.Tensor) *core.Tensor {
    // Custom backward pass
}
```

### Custom Optimizers
```go
type CustomOptimizer struct {
    LearningRate float64
}

func (c *CustomOptimizer) Update(params []*core.Tensor, gradients []*core.Tensor) {
    // Custom update rule
}
```

### Custom Callbacks
```go
type CustomCallback struct{}

func (c *CustomCallback) OnEpochBegin(epoch int, history map[string][]float64) {
    // Custom epoch begin logic
}

func (c *CustomCallback) OnEpochEnd(epoch int, history map[string][]float64) {
    // Custom epoch end logic
}
```

## 🚀 Performance

- **Pure Go**: No external dependencies, fast compilation
- **Memory Efficient**: Optimized tensor operations
- **Concurrent**: Leverages Go's goroutines for parallel processing
- **Production Ready**: Comprehensive error handling and logging

## 🤝 Contributing

We welcome contributions from developers of all skill levels! Here's how you can contribute:

### **Ways to Contribute**
- 🐛 **Bug Reports**: Found a bug? Report it using our issue templates
- ✨ **Feature Requests**: Have an idea? Submit a feature request
- 🔧 **Code Contributions**: Fix bugs, implement features, improve code
- 📚 **Documentation**: Improve docs, add tutorials, fix typos
- 🧪 **Testing**: Add tests, improve test coverage

### **Getting Started**
1. **Fork the repository** and clone your fork
2. **Read our [Contributing Guide](CONTRIBUTING.md)** for detailed instructions
3. **Check our [Developer Onboarding Guide](DEVELOPER_ONBOARDING.md)** for setup help
4. **Look for [good first issues](https://github.com/gointellect/gointellect/labels/good-first-issue)** to get started
5. **Join our [Discord community](https://discord.gg/gointellect)** for real-time help

### **Community Guidelines**
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)
- Be respectful and inclusive
- Help others learn and grow
- Focus on what's best for the community

### **Recognition**
- Contributors listed in README
- GitHub contributor graphs
- Release notes acknowledgment
- Contributor badges and certificates

## 📄 License

MIT License - see LICENSE file for details.

## 🎯 Roadmap

- [ ] Convolutional Neural Networks (CNN)
- [ ] Recurrent Neural Networks (RNN/LSTM)
- [ ] Transformer architecture
- [ ] Model serialization/deserialization
- [ ] GPU acceleration (CUDA support)
- [ ] Distributed training
- [ ] Model serving framework
- [ ] Visualization tools
- [ ] More traditional ML algorithms (SVM, Random Forest, etc.)

---

**gointellect** - Bringing the power of machine learning to the Go ecosystem! 🚀
# GOintellect
# GOintellect
