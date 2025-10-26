package train

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/gointellect/gointellect/pkg/core"
	"github.com/gointellect/gointellect/pkg/nn"
)

// Trainer handles the training process
type Trainer struct {
	Model     *nn.NeuralNetwork
	Loss      nn.LossFunction
	Optimizer nn.Optimizer
	Metrics   []Metric
	Callbacks []Callback
	Verbose   bool
}

// NewTrainer creates a new trainer
func NewTrainer(model *nn.NeuralNetwork, loss nn.LossFunction, optimizer nn.Optimizer) *Trainer {
	return &Trainer{
		Model:     model,
		Loss:      loss,
		Optimizer: optimizer,
		Metrics:   []Metric{},
		Callbacks: []Callback{},
		Verbose:   true,
	}
}

// AddMetric adds a metric to track during training
func (t *Trainer) AddMetric(metric Metric) {
	t.Metrics = append(t.Metrics, metric)
}

// AddCallback adds a callback for training events
func (t *Trainer) AddCallback(callback Callback) {
	t.Callbacks = append(t.Callbacks, callback)
}

// TrainConfig holds training configuration
type TrainConfig struct {
	Epochs          int
	BatchSize       int
	ValidationSplit float64
	Shuffle         bool
	EarlyStopping   EarlyStoppingConfig
}

// EarlyStoppingConfig holds early stopping configuration
type EarlyStoppingConfig struct {
	Enabled     bool
	Patience    int
	MinDelta    float64
	Monitor     string // "loss" or "val_loss"
	RestoreBest bool
}

// TrainResult holds training results
type TrainResult struct {
	History      map[string][]float64
	BestEpoch    int
	BestScore    float64
	TrainingTime time.Duration
}

// Train performs the training process
func (t *Trainer) Train(X, y *core.Tensor, config TrainConfig) *TrainResult {
	startTime := time.Now()

	// Initialize history
	history := make(map[string][]float64)
	history["loss"] = []float64{}
	history["val_loss"] = []float64{}

	for _, metric := range t.Metrics {
		history[metric.Name()] = []float64{}
		history["val_"+metric.Name()] = []float64{}
	}

	// Split data if validation split is specified
	var XTrain, yTrain, XVal, yVal *core.Tensor
	if config.ValidationSplit > 0 {
		XTrain, yTrain, XVal, yVal = t.splitData(X, y, config.ValidationSplit)
	} else {
		XTrain, yTrain = X, y
	}

	// Early stopping variables
	bestScore := math.Inf(1)
	bestEpoch := 0
	patienceCounter := 0

	// Training loop
	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochStart := time.Now()

		// Call on_epoch_begin callbacks
		for _, callback := range t.Callbacks {
			callback.OnEpochBegin(epoch, history)
		}

		// Training
		trainLoss := t.trainEpoch(XTrain, yTrain, config.BatchSize, config.Shuffle)
		history["loss"] = append(history["loss"], trainLoss)

		// Validation
		var valLoss float64
		if XVal != nil {
			valLoss = t.validate(XVal, yVal)
			history["val_loss"] = append(history["val_loss"], valLoss)
		}

		// Calculate metrics
		for _, metric := range t.Metrics {
			trainMetric := metric.Calculate(t.Model.Forward(XTrain), yTrain)
			history[metric.Name()] = append(history[metric.Name()], trainMetric)

			if XVal != nil {
				valMetric := metric.Calculate(t.Model.Forward(XVal), yVal)
				history["val_"+metric.Name()] = append(history["val_"+metric.Name()], valMetric)
			}
		}

		// Early stopping check
		if config.EarlyStopping.Enabled {
			monitorValue := trainLoss
			if config.EarlyStopping.Monitor == "val_loss" && XVal != nil {
				monitorValue = valLoss
			}

			if monitorValue < bestScore-config.EarlyStopping.MinDelta {
				bestScore = monitorValue
				bestEpoch = epoch
				patienceCounter = 0
			} else {
				patienceCounter++
			}

			if patienceCounter >= config.EarlyStopping.Patience {
				if t.Verbose {
					fmt.Printf("Early stopping at epoch %d\n", epoch)
				}
				break
			}
		}

		// Call on_epoch_end callbacks
		for _, callback := range t.Callbacks {
			callback.OnEpochEnd(epoch, history)
		}

		epochTime := time.Since(epochStart)
		if t.Verbose {
			fmt.Printf("Epoch %d/%d - loss: %.4f", epoch+1, config.Epochs, trainLoss)
			if XVal != nil {
				fmt.Printf(" - val_loss: %.4f", valLoss)
			}
			fmt.Printf(" - time: %v\n", epochTime)
		}
	}

	trainingTime := time.Since(startTime)

	return &TrainResult{
		History:      history,
		BestEpoch:    bestEpoch,
		BestScore:    bestScore,
		TrainingTime: trainingTime,
	}
}

// trainEpoch trains for one epoch
func (t *Trainer) trainEpoch(X, y *core.Tensor, batchSize int, shuffle bool) float64 {
	totalLoss := 0.0
	numBatches := 0

	// Create batches
	batches := t.createBatches(X, y, batchSize, shuffle)

	for _, batch := range batches {
		// Forward pass
		predictions := t.Model.Forward(batch.X)
		loss := t.Loss.Forward(predictions, batch.y)

		// Backward pass
		gradient := t.Loss.Backward(predictions, batch.y)
		t.Model.Backward(gradient)

		// Update parameters
		params := t.Model.Parameters()
		gradients := t.extractGradients(params)
		t.Optimizer.Update(params, gradients)

		totalLoss += loss
		numBatches++
	}

	return totalLoss / float64(numBatches)
}

// validate performs validation
func (t *Trainer) validate(X, y *core.Tensor) float64 {
	predictions := t.Model.Forward(X)
	return t.Loss.Forward(predictions, y)
}

// Batch represents a training batch
type Batch struct {
	X *core.Tensor
	y *core.Tensor
}

// createBatches creates training batches
func (t *Trainer) createBatches(X, y *core.Tensor, batchSize int, shuffle bool) []Batch {
	numSamples := X.Shape[0]
	numBatches := (numSamples + batchSize - 1) / batchSize

	batches := make([]Batch, numBatches)

	// Create indices
	indices := make([]int, numSamples)
	for i := range indices {
		indices[i] = i
	}
	// Shuffle if requested
	if shuffle {
		for i := len(indices) - 1; i > 0; i-- {
			j := rand.Intn(i + 1)
			indices[i], indices[j] = indices[j], indices[i]
		}
	}

	// Create batches
	for i := 0; i < numBatches; i++ {
		start := i * batchSize
		end := start + batchSize
		if end > numSamples {
			end = numSamples
		}

		bs := end - start
		batchX := core.NewTensor2D(bs, X.Shape[1])
		batchY := core.NewTensor2D(bs, y.Shape[1])

		for j := 0; j < bs; j++ {
			idx := indices[start+j]

			// Copy X data
			for k := 0; k < X.Shape[1]; k++ {
				batchX.Set(X.At(idx, k), j, k)
			}

			// Copy y data
			for k := 0; k < y.Shape[1]; k++ {
				batchY.Set(y.At(idx, k), j, k)
			}
		}

		batches[i] = Batch{X: batchX, y: batchY}
	}

	return batches
}

// splitData splits data into training and validation sets
func (t *Trainer) splitData(X, y *core.Tensor, validationSplit float64) (*core.Tensor, *core.Tensor, *core.Tensor, *core.Tensor) {
	numSamples := X.Shape[0]
	numValSamples := int(float64(numSamples) * validationSplit)
	numTrainSamples := numSamples - numValSamples

	// Create training set
	trainX := core.NewTensor2D(numTrainSamples, X.Shape[1])
	trainY := core.NewTensor2D(numTrainSamples, y.Shape[1])

	// Create validation set
	valX := core.NewTensor2D(numValSamples, X.Shape[1])
	valY := core.NewTensor2D(numValSamples, y.Shape[1])

	// Split data
	for i := 0; i < numTrainSamples; i++ {
		for j := 0; j < X.Shape[1]; j++ {
			trainX.Set(X.At(i, j), i, j)
		}
		for j := 0; j < y.Shape[1]; j++ {
			trainY.Set(y.At(i, j), i, j)
		}
	}

	for i := 0; i < numValSamples; i++ {
		idx := numTrainSamples + i
		for j := 0; j < X.Shape[1]; j++ {
			valX.Set(X.At(idx, j), i, j)
		}
		for j := 0; j < y.Shape[1]; j++ {
			valY.Set(y.At(idx, j), i, j)
		}
	}

	return trainX, trainY, valX, valY
}

// extractGradients extracts gradients from parameters (simplified)
func (t *Trainer) extractGradients(params []*core.Tensor) []*core.Tensor {
	// This is a simplified implementation
	// In a real implementation, gradients would be computed during backward pass
	gradients := make([]*core.Tensor, len(params))
	for i, param := range params {
		gradients[i] = core.Zeros(param.Shape)
	}
	return gradients
}

// Metric interface for evaluation metrics
type Metric interface {
	Name() string
	Calculate(predictions, targets *core.Tensor) float64
}

// Accuracy metric
type Accuracy struct{}

func NewAccuracy() *Accuracy {
	return &Accuracy{}
}

func (a *Accuracy) Name() string {
	return "accuracy"
}

func (a *Accuracy) Calculate(predictions, targets *core.Tensor) float64 {
	correct := 0
	total := predictions.Shape[0]

	for i := 0; i < total; i++ {
		// Find predicted class
		predClass := 0
		maxProb := predictions.At(i, 0)
		for j := 1; j < predictions.Shape[1]; j++ {
			if predictions.At(i, j) > maxProb {
				maxProb = predictions.At(i, j)
				predClass = j
			}
		}

		// Find true class
		trueClass := 0
		maxTarget := targets.At(i, 0)
		for j := 1; j < targets.Shape[1]; j++ {
			if targets.At(i, j) > maxTarget {
				maxTarget = targets.At(i, j)
				trueClass = j
			}
		}

		if predClass == trueClass {
			correct++
		}
	}

	return float64(correct) / float64(total)
}

// Callback interface for training callbacks
type Callback interface {
	OnEpochBegin(epoch int, history map[string][]float64)
	OnEpochEnd(epoch int, history map[string][]float64)
}

// LearningRateScheduler callback
type LearningRateScheduler struct {
	Schedule  func(epoch int) float64
	Optimizer nn.Optimizer
}

func NewLearningRateScheduler(schedule func(epoch int) float64, optimizer nn.Optimizer) *LearningRateScheduler {
	return &LearningRateScheduler{
		Schedule:  schedule,
		Optimizer: optimizer,
	}
}

func (lrs *LearningRateScheduler) OnEpochBegin(epoch int, history map[string][]float64) {
	if sgd, ok := lrs.Optimizer.(*nn.SGD); ok {
		sgd.LearningRate = lrs.Schedule(epoch)
	}
}

func (lrs *LearningRateScheduler) OnEpochEnd(epoch int, history map[string][]float64) {
	// No action needed
}

// ModelCheckpoint callback
type ModelCheckpoint struct {
	FilePath  string
	Monitor   string
	SaveBest  bool
	BestScore float64
}

func NewModelCheckpoint(filePath, monitor string, saveBest bool) *ModelCheckpoint {
	return &ModelCheckpoint{
		FilePath:  filePath,
		Monitor:   monitor,
		SaveBest:  saveBest,
		BestScore: math.Inf(1),
	}
}

func (mc *ModelCheckpoint) OnEpochBegin(epoch int, history map[string][]float64) {
	// No action needed
}

func (mc *ModelCheckpoint) OnEpochEnd(epoch int, history map[string][]float64) {
	if mc.SaveBest {
		if values, exists := history[mc.Monitor]; exists && len(values) > 0 {
			currentScore := values[len(values)-1]
			if currentScore < mc.BestScore {
				mc.BestScore = currentScore
				// In a real implementation, save the model here
				fmt.Printf("New best %s: %.4f, saving model to %s\n", mc.Monitor, currentScore, mc.FilePath)
			}
		}
	}
}
