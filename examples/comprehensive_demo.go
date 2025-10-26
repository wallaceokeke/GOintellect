package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/wallaceokeke/GOintellect/pkg/core"
	"github.com/wallaceokeke/GOintellect/pkg/data"
	"github.com/wallaceokeke/GOintellect/pkg/nn"
	"github.com/wallaceokeke/GOintellect/pkg/train"
)

func RunComprehensiveDemo() {
	// Initialize random seed
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== gointellect: Comprehensive ML Library Demo ===")

	// 1. Tensor Operations Demo
	fmt.Println("1. Tensor Operations:")
	demonstrateTensorOps()

	// 2. Data Pipeline Demo
	fmt.Println("\n2. Data Pipeline:")
	demonstrateDataPipeline()

	// 3. Neural Network Demo
	fmt.Println("\n3. Neural Network Training:")
	demonstrateNeuralNetwork()

	// 4. Model Evaluation Demo
	fmt.Println("\n4. Model Evaluation:")
	demonstrateModelEvaluation()
}

func demonstrateTensorOps() {
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

	fmt.Printf("Tensor A:\n%v\n", a)
	fmt.Printf("Tensor B:\n%v\n", b)

	// Element-wise operations
	c := a.Add(b)
	fmt.Printf("A + B:\n%v\n", c)

	d := a.Mul(b)
	fmt.Printf("A * B:\n%v\n", d)

	// Matrix multiplication
	e := core.NewTensor2D(3, 2)
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			e.Set(float64(i*2+j+1), i, j)
		}
	}

	f := a.Dot(e)
	fmt.Printf("A @ E:\n%v\n", f)

	// Statistics
	fmt.Printf("Sum of A: %.2f\n", a.Sum())
	fmt.Printf("Mean of A: %.2f\n", a.Mean())
}

func demonstrateDataPipeline() {
	// Create synthetic dataset
	numSamples := 100
	numFeatures := 4

	features := core.NewTensor2D(numSamples, numFeatures)
	targets := core.NewTensor2D(numSamples, 1)

	// Generate synthetic data: y = 2*x1 + 3*x2 - x3 + 0.5*x4 + noise
	for i := 0; i < numSamples; i++ {
		x1 := rand.Float64() * 10
		x2 := rand.Float64() * 10
		x3 := rand.Float64() * 10
		x4 := rand.Float64() * 10

		features.Set(x1, i, 0)
		features.Set(x2, i, 1)
		features.Set(x3, i, 2)
		features.Set(x4, i, 3)

		y := 2*x1 + 3*x2 - x3 + 0.5*x4 + (rand.Float64()-0.5)*2
		targets.Set(y, i, 0)
	}

	dataset := data.NewDataset(features, targets)
	fmt.Printf("Original dataset shape: %v x %v\n", dataset.Features.Shape, dataset.Targets.Shape)

	// Apply standardization
	scaler := data.NewStandardScaler()
	scaledDataset, err := scaler.FitTransform(dataset)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Scaled dataset shape: %v x %v\n", scaledDataset.Features.Shape, scaledDataset.Targets.Shape)
	fmt.Printf("Scaled features mean: %.4f\n", scaledDataset.Features.Mean())

	// Split data
	splitter := data.NewTrainTestSplitter(0.2)
	splits, err := splitter.Split(scaledDataset)
	if err != nil {
		log.Fatal(err)
	}

	trainData := splits[0]
	testData := splits[1]

	fmt.Printf("Train set: %v samples\n", trainData.Features.Shape[0])
	fmt.Printf("Test set: %v samples\n", testData.Features.Shape[0])
}

func demonstrateNeuralNetwork() {
	// Create synthetic classification dataset
	numSamples := 200
	numFeatures := 2

	features := core.NewTensor2D(numSamples, numFeatures)
	targets := core.NewTensor2D(numSamples, 2) // One-hot encoded

	// Generate XOR-like dataset
	for i := 0; i < numSamples; i++ {
		x1 := rand.Float64()*4 - 2
		x2 := rand.Float64()*4 - 2

		features.Set(x1, i, 0)
		features.Set(x2, i, 1)

		// XOR-like pattern
		if (x1 > 0 && x2 > 0) || (x1 < 0 && x2 < 0) {
			targets.Set(1, i, 0) // Class 0
			targets.Set(0, i, 1)
		} else {
			targets.Set(0, i, 0) // Class 1
			targets.Set(1, i, 1)
		}
	}

	dataset := data.NewDataset(features, targets)

	// Split data
	splitter := data.NewTrainTestSplitter(0.3)
	splits, err := splitter.Split(dataset)
	if err != nil {
		log.Fatal(err)
	}

	trainData := splits[0]
	testData := splits[1]

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
		Epochs:          100,
		BatchSize:       32,
		ValidationSplit: 0.2,
		Shuffle:         true,
		EarlyStopping: train.EarlyStoppingConfig{
			Enabled:  true,
			Patience: 10,
			MinDelta: 0.001,
			Monitor:  "val_loss",
		},
	}

	// Train the model
	fmt.Println("Training neural network...")
	result := trainer.Train(trainData.Features, trainData.Targets, config)

	fmt.Printf("Training completed in %v\n", result.TrainingTime)
	fmt.Printf("Best epoch: %d with score: %.4f\n", result.BestEpoch, result.BestScore)

	// Test the model
	predictions := model.Forward(testData.Features)
	testLoss := loss.Forward(predictions, testData.Targets)

	fmt.Printf("Test loss: %.4f\n", testLoss)

	// Calculate accuracy
	accuracy := train.NewAccuracy()
	testAccuracy := accuracy.Calculate(predictions, testData.Targets)
	fmt.Printf("Test accuracy: %.4f\n", testAccuracy)
}

func demonstrateModelEvaluation() {
	// Create a simple regression model
	features := core.NewTensor2D(50, 1)
	targets := core.NewTensor2D(50, 1)

	// Generate linear relationship with noise
	for i := 0; i < 50; i++ {
		x := float64(i) / 10.0
		y := 2*x + 1 + (rand.Float64()-0.5)*0.5

		features.Set(x, i, 0)
		targets.Set(y, i, 0)
	}

	// Create simple linear model (single neuron)
	model := nn.NewNeuralNetwork(
		nn.NewDenseLayer(1, 1, nil), // No activation for regression
	)

	// Train with MSE loss
	loss := nn.NewMeanSquaredError()
	optimizer := nn.NewSGD(0.01)
	trainer := train.NewTrainer(model, loss, optimizer)

	config := train.TrainConfig{
		Epochs:    50,
		BatchSize: 10,
		Shuffle:   true,
	}

	fmt.Println("Training regression model...")
	result := trainer.Train(features, targets, config)

	fmt.Printf("Final training loss: %.4f\n", result.History["loss"][len(result.History["loss"])-1])

	// Make predictions
	predictions := model.Forward(features)

	// Calculate R² score
	r2 := calculateR2(predictions, targets)
	fmt.Printf("R² Score: %.4f\n", r2)

	// Show some predictions
	fmt.Println("Sample predictions:")
	for i := 0; i < 5; i++ {
		actual := targets.At(i, 0)
		predicted := predictions.At(i, 0)
		fmt.Printf("  x=%.2f, actual=%.2f, predicted=%.2f\n",
			features.At(i, 0), actual, predicted)
	}
}

// calculateR2 calculates the R² score
func calculateR2(predictions, targets *core.Tensor) float64 {
	// Calculate mean of targets
	targetMean := targets.Mean()

	// Calculate SS_res and SS_tot
	ssRes := 0.0
	ssTot := 0.0

	for i := 0; i < predictions.Size(); i++ {
		residual := predictions.Data[i] - targets.Data[i]
		ssRes += residual * residual

		total := targets.Data[i] - targetMean
		ssTot += total * total
	}

	if ssTot == 0 {
		return 0
	}

	return 1 - (ssRes / ssTot)
}
