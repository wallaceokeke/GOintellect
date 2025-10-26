package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/wallaceokeke/GOintellect/pkg/automl"
	"github.com/wallaceokeke/GOintellect/pkg/core"
	"github.com/wallaceokeke/GOintellect/pkg/model"
	"github.com/wallaceokeke/GOintellect/pkg/nn"
	"github.com/wallaceokeke/GOintellect/pkg/serving"
	"github.com/wallaceokeke/GOintellect/pkg/train"
)

func RunDemos() {
	// Initialize random seed
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== gointellect: STEROIDS-LEVEL ML Library Demo ===")

	// 1. Advanced Tensor Operations
	fmt.Println("1. Advanced Tensor Operations:")
	demonstrateAdvancedTensorOps()

	// 2. Convolutional Neural Networks
	fmt.Println("\n2. Convolutional Neural Networks:")
	demonstrateCNN()

	// 3. Recurrent Neural Networks
	fmt.Println("\n3. Recurrent Neural Networks:")
	demonstrateRNN()

	// 4. Advanced Optimizers and Regularization
	fmt.Println("\n4. Advanced Optimizers and Regularization:")
	demonstrateAdvancedOptimizers()

	// 5. Model Serialization and Persistence
	fmt.Println("\n5. Model Serialization and Persistence:")
	demonstrateModelSerialization()

	// 6. Model Serving Framework
	fmt.Println("\n6. Model Serving Framework:")
	demonstrateModelServing()

	// 7. AutoML and Hyperparameter Optimization
	fmt.Println("\n7. AutoML and Hyperparameter Optimization:")
	demonstrateAutoML()

	fmt.Println("\n=== STEROIDS-LEVEL DEMO COMPLETE ===")
}

func demonstrateAdvancedTensorOps() {
	// Create complex tensors
	a := core.NewTensor([]int{2, 3, 4})
	b := core.NewTensor([]int{2, 3, 4})

	// Fill with data
	for i := 0; i < a.Size(); i++ {
		a.Data[i] = float64(i)
		b.Data[i] = float64(i * 2)
	}

	fmt.Printf("Tensor A shape: %v\n", a.Shape)
	fmt.Printf("Tensor B shape: %v\n", b.Shape)

	// Element-wise operations
	c := a.Add(b)
	fmt.Printf("A + B sum: %.2f\n", c.Sum())

	// Reshape operations
	d := a.Reshape([]int{6, 4})
	fmt.Printf("Reshaped tensor shape: %v\n", d.Shape)

	// Broadcasting simulation
	e := core.NewTensor2D(3, 4)
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			e.Set(float64(i+j), i, j)
		}
	}

	f := e.Scale(2.0)
	fmt.Printf("Scaled tensor mean: %.2f\n", f.Mean())
}

func demonstrateCNN() {
	// Create synthetic image data [batch, channels, height, width]
	batchSize := 4
	channels := 3
	height := 32
	width := 32

	// Create input tensor
	input := core.NewTensor([]int{batchSize, channels, height, width})
	for i := 0; i < input.Size(); i++ {
		input.Data[i] = rand.Float64()*2 - 1 // Random values between -1 and 1
	}

	fmt.Printf("Input shape: %v\n", input.Shape)

	// Create CNN layers
	conv1 := nn.NewConv2DLayer(32, [2]int{3, 3}, [2]int{1, 1}, [2]int{1, 1}, nn.NewReLU())
	pool1 := nn.NewMaxPool2DLayer([2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0})
	conv2 := nn.NewConv2DLayer(64, [2]int{3, 3}, [2]int{1, 1}, [2]int{1, 1}, nn.NewReLU())
	pool2 := nn.NewMaxPool2DLayer([2]int{2, 2}, [2]int{2, 2}, [2]int{0, 0})
	flatten := nn.NewFlattenLayer()
	dense1 := nn.NewDenseLayer(1024, 128, nn.NewReLU())
	dropout := nn.NewDropoutLayer(0.5)
	dense2 := nn.NewDenseLayer(128, 10, nn.NewSoftmax())

	// Create CNN model
	cnn := nn.NewNeuralNetwork(conv1, pool1, conv2, pool2, flatten, dense1, dropout, dense2)

	// Forward pass
	output := cnn.Forward(input)
	fmt.Printf("CNN output shape: %v\n", output.Shape)
	fmt.Printf("CNN output sample: %v\n", output.Data[:10])
}

func demonstrateRNN() {
	// Create synthetic sequence data [batch, sequence, features]
	batchSize := 2
	seqLen := 10
	features := 5

	input := core.NewTensor([]int{batchSize, seqLen, features})
	for i := 0; i < input.Size(); i++ {
		input.Data[i] = rand.Float64()*2 - 1
	}

	fmt.Printf("RNN input shape: %v\n", input.Shape)

	// Create RNN layers
	rnn1 := nn.NewRNNLayer(features, 32, nn.NewTanh())
	lstm1 := nn.NewLSTMLayer(32, 64)
	gru1 := nn.NewGRULayer(64, 32)
	dense := nn.NewDenseLayer(32, 2, nn.NewSoftmax())

	// Create RNN model
	rnn := nn.NewNeuralNetwork(rnn1, lstm1, gru1, dense)

	// Forward pass
	output := rnn.Forward(input)
	fmt.Printf("RNN output shape: %v\n", output.Shape)
	fmt.Printf("RNN output sample: %v\n", output.Data[:10])
}

func demonstrateAdvancedOptimizers() {
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

	// Test different optimizers
	optimizers := map[string]nn.Optimizer{
		"SGD":      nn.NewSGD(0.01),
		"Adam":     nn.NewAdam(0.001),
		"RMSprop":  nn.NewRMSprop(0.001),
		"AdaGrad":  nn.NewAdaGrad(0.01),
		"AdaDelta": nn.NewAdaDelta(),
		"Nadam":    nn.NewNadam(0.001),
	}

	for name, optimizer := range optimizers {
		// Create model
		model := nn.NewNeuralNetwork(
			nn.NewDenseLayer(4, 16, nn.NewReLU()),
			nn.NewDropoutLayer(0.2),
			nn.NewBatchNormLayer(16),
			nn.NewDenseLayer(16, 8, nn.NewReLU()),
			nn.NewDenseLayer(8, 2, nn.NewSoftmax()),
		)

		// Create trainer
		loss := nn.NewCrossEntropy()
		trainer := train.NewTrainer(model, loss, optimizer)
		trainer.AddMetric(train.NewAccuracy())

		// Train
		config := train.TrainConfig{
			Epochs:    20,
			BatchSize: 16,
			Shuffle:   true,
		}

		result := trainer.Train(X, y, config)
		finalAccuracy := result.History["accuracy"][len(result.History["accuracy"])-1]

		fmt.Printf("%s optimizer - Final accuracy: %.4f\n", name, finalAccuracy)
	}

	// Test regularization
	fmt.Println("\nTesting regularization:")

	// L1 Regularization
	l1Reg := nn.NewL1Regularizer(0.01)
	penalty := l1Reg.Regularize([]*core.Tensor{X})
	fmt.Printf("L1 regularization penalty: %.4f\n", penalty)

	// L2 Regularization
	l2Reg := nn.NewL2Regularizer(0.01)
	penalty = l2Reg.Regularize([]*core.Tensor{X})
	fmt.Printf("L2 regularization penalty: %.4f\n", penalty)

	// Elastic Net Regularization
	elasticNet := nn.NewElasticNetRegularizer(0.01, 0.01)
	penalty = elasticNet.Regularize([]*core.Tensor{X})
	fmt.Printf("Elastic Net regularization penalty: %.4f\n", penalty)
}

func demonstrateModelSerialization() {
	// Create a model
	nnModel := nn.NewNeuralNetwork(
		nn.NewDenseLayer(4, 16, nn.NewReLU()),
		nn.NewDropoutLayer(0.2),
		nn.NewDenseLayer(16, 8, nn.NewReLU()),
		nn.NewDenseLayer(8, 2, nn.NewSoftmax()),
	)

	// Create model manager
	manager := model.NewModelManager("./models")

	// Create metadata
	metadata := model.ModelMetadata{
		Name:         "demo_model",
		Version:      "1.0.0",
		CreatedAt:    time.Now(),
		Description:  "Demo model for serialization",
		Architecture: "neural_network",
		// CountParameters is unexported in model.ModelManager; use a placeholder value here
		// or export the method in the model package and call it instead.
		Parameters:  0,
		InputShape:  []int{4},
		OutputShape: []int{2},
		Tags:        []string{"demo", "classification"},
		Metrics: map[string]float64{
			"accuracy": 0.95,
			"loss":     0.1,
		},
		Config: map[string]interface{}{
			"optimizer":     "adam",
			"learning_rate": 0.001,
		},
	}
	_ = metadata

	// Save model
	err := manager.SaveModel(nnModel, "demo_model", "Demo model for serialization")
	if err != nil {
		log.Printf("Error saving model: %v", err)
	} else {
		fmt.Println("Model saved successfully")
	}

	// Load model
	_, loadedMetadata, err := manager.LoadModel("demo_model")
	if err != nil {
		log.Printf("Error loading model: %v", err)
	} else {
		fmt.Printf("Model loaded successfully: %s\n", loadedMetadata.Name)
		fmt.Printf("Model parameters: %d\n", loadedMetadata.Parameters)
	}

	// List models
	models, err := manager.ListModels()
	if err != nil {
		log.Printf("Error listing models: %v", err)
	} else {
		fmt.Printf("Available models: %d\n", len(models))
		for _, m := range models {
			fmt.Printf("  - %s (v%s): %s\n", m.Name, m.Version, m.Description)
		}
	}
}

func demonstrateModelServing() {
	// Create a simple model
	model := nn.NewNeuralNetwork(
		nn.NewDenseLayer(4, 8, nn.NewReLU()),
		nn.NewDenseLayer(8, 2, nn.NewSoftmax()),
	)
	_ = model

	// Create model server
	server := serving.NewModelServer(8080, "./models")

	// Load model
	err := server.LoadModel("demo_model")
	if err != nil {
		log.Printf("Error loading model: %v", err)
	}

	// Create model client
	client := serving.NewModelClient("http://localhost:8080")

	// Test prediction
	input := [][]float64{
		{1.0, 2.0, 3.0, 4.0},
		{5.0, 6.0, 7.0, 8.0},
	}

	prediction, err := client.Predict(input, "demo_model")
	if err != nil {
		log.Printf("Error making prediction: %v", err)
	} else {
		fmt.Printf("Prediction result: %v\n", prediction.Predictions)
		fmt.Printf("Latency: %dms\n", prediction.Latency)
	}

	// Health check
	health, err := client.HealthCheck()
	if err != nil {
		log.Printf("Error checking health: %v", err)
	} else {
		fmt.Printf("Server health: %v\n", health["status"])
		fmt.Printf("Models loaded: %v\n", health["models"])
	}
}

func demonstrateAutoML() {
	// Create synthetic dataset
	X := core.NewTensor2D(200, 4)
	y := core.NewTensor2D(200, 2)

	for i := 0; i < 200; i++ {
		for j := 0; j < 4; j++ {
			X.Set(rand.Float64()*10, i, j)
		}

		// Create classification targets
		if i < 100 {
			y.Set(1, i, 0)
			y.Set(0, i, 1)
		} else {
			y.Set(0, i, 0)
			y.Set(1, i, 1)
		}
	}

	// Create AutoML configuration
	config := automl.DefaultAutoMLConfig()
	config.MaxTrials = 10
	config.MaxTime = 5 * time.Minute
	config.SearchStrategy = "random"
	config.ScoringMetric = "accuracy"

	// Create search space
	searchSpace := automl.DefaultSearchSpace()

	// Create AutoML instance
	// Provide a TrainerFunc (nil placeholder) to satisfy the constructor signature;
	// replace with a real trainer implementation as needed.
	autoML := automl.NewAutoML(config, searchSpace, automl.TrainerFunc(nil))

	// Run AutoML
	fmt.Println("Running AutoML...")
	bestResult, err := autoML.Search(X, y)
	if err != nil {
		log.Printf("Error running AutoML: %v", err)
		return
	}

	fmt.Printf("Best configuration found:\n")
	fmt.Printf("  Learning Rate: %.4f\n", bestResult.Config.LearningRate)
	fmt.Printf("  Batch Size: %d\n", bestResult.Config.BatchSize)
	fmt.Printf("  Optimizer: %s\n", bestResult.Config.Optimizer)
	fmt.Printf("  Epochs: %d\n", bestResult.Config.Epochs)
	fmt.Printf("  Hidden Sizes: %v\n", bestResult.Config.HiddenSizes)
	fmt.Printf("  Dropout Rates: %v\n", bestResult.Config.DropoutRates)
	fmt.Printf("  Activation Functions: %v\n", bestResult.Config.ActivationFuncs)
	fmt.Printf("  Best Score: %.4f\n", bestResult.Score)
	fmt.Printf("  Best Accuracy: %.4f\n", bestResult.Accuracy)
	fmt.Printf("  Training Time: %v\n", bestResult.Time)

	// Test Bayesian optimization
	fmt.Println("\nTesting Bayesian Optimization...")
	bayesianOpt := automl.NewBayesianOptimizer(searchSpace)

	for i := 0; i < 5; i++ {
		config := bayesianOpt.SuggestNextParameters()
		fmt.Printf("Bayesian suggestion %d: LR=%.4f, Batch=%d, Optimizer=%s\n",
			i+1, config.LearningRate, config.BatchSize, config.Optimizer)

		// Simulate a result
		result := automl.TrialResult{
			Config: config,
			Score:  rand.Float64(),
			Status: "success",
		}
		bayesianOpt.AddResult(result)
	}
}
