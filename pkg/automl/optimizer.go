package automl

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/wallaceokeke/GOintellect/pkg/core"
)

// HyperparameterSearchSpace defines the search space for hyperparameters
type HyperparameterSearchSpace struct {
	LearningRate    []float64 `json:"learning_rate"`
	BatchSize       []int     `json:"batch_size"`
	HiddenSizes     []int     `json:"hidden_sizes"`
	DropoutRates    []float64 `json:"dropout_rates"`
	ActivationFuncs []string  `json:"activation_funcs"`
	Optimizers      []string  `json:"optimizers"`
	Epochs          []int     `json:"epochs"`
}

// DefaultSearchSpace returns a default hyperparameter search space
func DefaultSearchSpace() *HyperparameterSearchSpace {
	return &HyperparameterSearchSpace{
		LearningRate:    []float64{0.001, 0.01, 0.1},
		BatchSize:       []int{16, 32, 64, 128},
		HiddenSizes:     []int{32, 64, 128, 256},
		DropoutRates:    []float64{0.0, 0.2, 0.5},
		ActivationFuncs: []string{"relu", "sigmoid", "tanh"},
		Optimizers:      []string{"sgd", "adam", "rmsprop"},
		Epochs:          []int{50, 100, 200},
	}
}

// HyperparameterConfig represents a specific hyperparameter configuration
type HyperparameterConfig struct {
	LearningRate    float64   `json:"learning_rate"`
	BatchSize       int       `json:"batch_size"`
	HiddenSizes     []int     `json:"hidden_sizes"`
	DropoutRates    []float64 `json:"dropout_rates"`
	ActivationFuncs []string  `json:"activation_funcs"`
	Optimizer       string    `json:"optimizer"`
	Epochs          int       `json:"epochs"`
}

// TrialResult represents the result of a hyperparameter trial
type TrialResult struct {
	Config   HyperparameterConfig `json:"config"`
	Score    float64              `json:"score"`
	Loss     float64              `json:"loss"`
	Accuracy float64              `json:"accuracy"`
	Epochs   int                  `json:"epochs"`
	Time     time.Duration        `json:"time"`
	Status   string               `json:"status"` // "success", "failed", "timeout"
	Error    string               `json:"error,omitempty"`
}

// AutoMLConfig holds configuration for AutoML
type AutoMLConfig struct {
	MaxTrials       int           `json:"max_trials"`
	MaxTime         time.Duration `json:"max_time"`
	SearchStrategy  string        `json:"search_strategy"` // "random", "grid", "bayesian"
	EarlyStopping   bool          `json:"early_stopping"`
	ValidationSplit float64       `json:"validation_split"`
	ScoringMetric   string        `json:"scoring_metric"` // "accuracy", "loss", "val_accuracy", "val_loss"
	ParallelTrials  int           `json:"parallel_trials"`
	RandomSeed      int64         `json:"random_seed"`
}

// DefaultAutoMLConfig returns a default AutoML configuration
func DefaultAutoMLConfig() *AutoMLConfig {
	return &AutoMLConfig{
		MaxTrials:       50,
		MaxTime:         2 * time.Hour,
		SearchStrategy:  "random",
		EarlyStopping:   true,
		ValidationSplit: 0.2,
		ScoringMetric:   "accuracy",
		ParallelTrials:  1,
		RandomSeed:      time.Now().UnixNano(),
	}
}

// TrainerFunc is a callback used by AutoML to execute a training trial.
// It receives the hyperparameter config and the dataset (X,y) and must return a TrialResult.
// If Trainer is nil AutoML will run a simple simulated training (useful for testing).
type TrainerFunc func(config HyperparameterConfig, X, y *core.Tensor) (TrialResult, error)

// AutoML handles automated machine learning
type AutoML struct {
	Config      *AutoMLConfig
	SearchSpace *HyperparameterSearchSpace
	Results     []TrialResult
	BestResult  *TrialResult
	StartTime   time.Time
	Trainer     TrainerFunc
}

// NewAutoML creates a new AutoML instance. Trainer may be nil; in that case Search will simulate trials.
func NewAutoML(config *AutoMLConfig, searchSpace *HyperparameterSearchSpace, trainer TrainerFunc) *AutoML {
	if config == nil {
		config = DefaultAutoMLConfig()
	}
	if config.RandomSeed == 0 {
		config.RandomSeed = time.Now().UnixNano()
	}
	if searchSpace == nil {
		searchSpace = DefaultSearchSpace()
	}
	return &AutoML{
		Config:      config,
		SearchSpace: searchSpace,
		Results:     []TrialResult{},
		StartTime:   time.Now(),
		Trainer:     trainer,
	}
}

// Search performs hyperparameter search
func (a *AutoML) Search(X, y *core.Tensor) (*TrialResult, error) {
	rand.Seed(a.Config.RandomSeed)

	// Basic sanity checks for provided tensors
	if X == nil || y == nil {
		return nil, fmt.Errorf("X and y must be provided")
	}
	if len(X.Shape) < 2 || len(y.Shape) < 2 {
		return nil, fmt.Errorf("X and y must have at least 2 dimensions (batch, features/classes)")
	}

	for len(a.Results) < a.Config.MaxTrials {
		// Check time limit
		if time.Since(a.StartTime) > a.Config.MaxTime {
			break
		}

		// Generate hyperparameter configuration
		config := a.generateConfig()

		// Run trial
		result, err := a.runTrial(config, X, y)
		if err != nil {
			// record failed trial
			result.Status = "failed"
			result.Error = err.Error()
			a.Results = append(a.Results, result)
			fmt.Printf("Trial %d failed: %v\n", len(a.Results), err)
			continue
		}

		a.Results = append(a.Results, result)

		// Update best result
		if a.BestResult == nil || result.Score > a.BestResult.Score {
			a.BestResult = &result
		}

		fmt.Printf("Trial %d: Score=%.4f, Best=%.4f\n", len(a.Results), result.Score, a.BestResult.Score)
	}

	return a.BestResult, nil
}

// generateConfig generates a random hyperparameter configuration
func (a *AutoML) generateConfig() HyperparameterConfig {
	space := a.SearchSpace

	// safe random selection helpers
	ri := func(valsLen int) int {
		if valsLen == 0 {
			return 0
		}
		return rand.Intn(valsLen)
	}

	config := HyperparameterConfig{
		LearningRate: space.LearningRate[ri(len(space.LearningRate))],
		BatchSize:    space.BatchSize[ri(len(space.BatchSize))],
		Optimizer:    space.Optimizers[ri(len(space.Optimizers))],
		Epochs:       space.Epochs[ri(len(space.Epochs))],
	}

	// Generate hidden sizes (2-4 layers)
	numLayers := 2 + rand.Intn(3)
	if numLayers <= 0 {
		numLayers = 2
	}
	config.HiddenSizes = make([]int, numLayers)
	for i := 0; i < numLayers; i++ {
		config.HiddenSizes[i] = space.HiddenSizes[ri(len(space.HiddenSizes))]
	}

	// Generate dropout rates
	config.DropoutRates = make([]float64, numLayers)
	for i := 0; i < numLayers; i++ {
		config.DropoutRates[i] = space.DropoutRates[ri(len(space.DropoutRates))]
	}

	// Generate activation functions
	config.ActivationFuncs = make([]string, numLayers)
	for i := 0; i < numLayers; i++ {
		config.ActivationFuncs[i] = space.ActivationFuncs[ri(len(space.ActivationFuncs))]
	}

	return config
}

// runTrial runs a single hyperparameter trial, delegating to Trainer if provided,
// otherwise performing a simple simulated training to produce a TrialResult.
func (a *AutoML) runTrial(config HyperparameterConfig, X, y *core.Tensor) (TrialResult, error) {
	start := time.Now()

	// If user provided a TrainerFunc use it
	if a.Trainer != nil {
		res, err := a.Trainer(config, X, y)
		// ensure time and epochs set
		if res.Time == 0 {
			res.Time = time.Since(start)
		}
		if res.Epochs == 0 {
			res.Epochs = config.Epochs
		}
		// calculate score if not set
		if res.Score == 0 {
			res.Score = a.calculateScoreFromResult(&res)
		}
		return res, err
	}

	// Simulate training: produce plausible loss/accuracy curves without external deps.
	epochs := config.Epochs
	if epochs <= 0 {
		epochs = 50
	}

	// Simulate loss decreasing and accuracy increasing
	baseLoss := 1.0 / (config.LearningRate + 1e-6)
	loss := baseLoss
	var lastLoss float64
	var lastAcc float64
	for e := 0; e < epochs; e++ {
		decay := 0.9 + rand.Float64()*0.1
		loss = loss * decay
		accuracy := math.Min(1.0, 1.0-math.Exp(-float64(e+1)/float64(epochs))*0.9+rand.NormFloat64()*0.01)
		lastLoss = loss
		lastAcc = accuracy

		// simple early stopping simulation
		if a.Config.EarlyStopping && e > 5 {
			// if change is tiny, break
			if math.Abs(loss-lastLoss) < 1e-6 {
				break
			}
		}
	}

	res := TrialResult{
		Config:   config,
		Loss:     lastLoss,
		Accuracy: lastAcc,
		Epochs:   epochs,
		Time:     time.Since(start),
		Status:   "success",
	}
	res.Score = a.calculateScoreFromResult(&res)
	return res, nil
}

// calculateScoreFromResult computes a numeric score from TrialResult using configured metric.
func (a *AutoML) calculateScoreFromResult(res *TrialResult) float64 {
	switch a.Config.ScoringMetric {
	case "accuracy":
		return res.Accuracy
	case "loss":
		// negative because we maximize score
		return -res.Loss
	case "val_accuracy":
		// no separate validation in simulation; fallback to accuracy
		return res.Accuracy
	case "val_loss":
		return -res.Loss
	default:
		return res.Accuracy
	}
}

// BayesianOptimizer implements a lightweight Bayesian-like optimizer for hyperparameter search
type BayesianOptimizer struct {
	SearchSpace *HyperparameterSearchSpace
	Results     []TrialResult
	Acquisition string // "ei", "pi", "ucb"
}

// NewBayesianOptimizer creates a new Bayesian optimizer
func NewBayesianOptimizer(searchSpace *HyperparameterSearchSpace) *BayesianOptimizer {
	if searchSpace == nil {
		searchSpace = DefaultSearchSpace()
	}
	return &BayesianOptimizer{
		SearchSpace: searchSpace,
		Results:     []TrialResult{},
		Acquisition: "ei", // Expected Improvement
	}
}

// SuggestNextParameters suggests the next hyperparameters to try
func (b *BayesianOptimizer) SuggestNextParameters() HyperparameterConfig {
	if len(b.Results) < 5 {
		// Use random search for the first few trials
		return b.randomConfig()
	}

	// Use Bayesian-like heuristic
	return b.bayesianConfig()
}

// randomConfig generates a random configuration
func (b *BayesianOptimizer) randomConfig() HyperparameterConfig {
	space := b.SearchSpace
	ri := func(valsLen int) int {
		if valsLen == 0 {
			return 0
		}
		return rand.Intn(valsLen)
	}

	config := HyperparameterConfig{
		LearningRate: space.LearningRate[ri(len(space.LearningRate))],
		BatchSize:    space.BatchSize[ri(len(space.BatchSize))],
		Optimizer:    space.Optimizers[ri(len(space.Optimizers))],
		Epochs:       space.Epochs[ri(len(space.Epochs))],
	}

	numLayers := 2 + rand.Intn(3)
	if numLayers <= 0 {
		numLayers = 2
	}
	config.HiddenSizes = make([]int, numLayers)
	config.DropoutRates = make([]float64, numLayers)
	config.ActivationFuncs = make([]string, numLayers)

	for i := 0; i < numLayers; i++ {
		config.HiddenSizes[i] = space.HiddenSizes[ri(len(space.HiddenSizes))]
		config.DropoutRates[i] = space.DropoutRates[ri(len(space.DropoutRates))]
		config.ActivationFuncs[i] = space.ActivationFuncs[ri(len(space.ActivationFuncs))]
	}

	return config
}

// bayesianConfig generates configuration using a simple heuristic driven by best results
func (b *BayesianOptimizer) bayesianConfig() HyperparameterConfig {
	bestConfigs := b.getBestConfigs(3)

	config := HyperparameterConfig{
		LearningRate: b.interpolateLearningRate(bestConfigs),
		BatchSize:    b.interpolateBatchSize(bestConfigs),
		Optimizer:    b.mostCommonOptimizer(bestConfigs),
		Epochs:       b.interpolateEpochs(bestConfigs),
	}

	// Fill other params randomly from search space
	numLayers := 2 + rand.Intn(3)
	if numLayers <= 0 {
		numLayers = 2
	}
	config.HiddenSizes = make([]int, numLayers)
	config.DropoutRates = make([]float64, numLayers)
	config.ActivationFuncs = make([]string, numLayers)

	for i := 0; i < numLayers; i++ {
		config.HiddenSizes[i] = b.SearchSpace.HiddenSizes[rand.Intn(len(b.SearchSpace.HiddenSizes))]
		config.DropoutRates[i] = b.SearchSpace.DropoutRates[rand.Intn(len(b.SearchSpace.DropoutRates))]
		config.ActivationFuncs[i] = b.SearchSpace.ActivationFuncs[rand.Intn(len(b.SearchSpace.ActivationFuncs))]
	}

	return config
}

// getBestConfigs returns the best performing configurations (by score)
func (b *BayesianOptimizer) getBestConfigs(n int) []TrialResult {
	if len(b.Results) == 0 {
		return []TrialResult{}
	}

	sorted := make([]TrialResult, len(b.Results))
	copy(sorted, b.Results)

	// simple descending sort by Score (selection sort)
	for i := 0; i < len(sorted); i++ {
		bestIdx := i
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].Score > sorted[bestIdx].Score {
				bestIdx = j
			}
		}
		if bestIdx != i {
			sorted[i], sorted[bestIdx] = sorted[bestIdx], sorted[i]
		}
	}

	if n > len(sorted) {
		n = len(sorted)
	}
	return sorted[:n]
}

// interpolateLearningRate interpolates learning rate from best configs
func (b *BayesianOptimizer) interpolateLearningRate(bestConfigs []TrialResult) float64 {
	if len(bestConfigs) == 0 {
		return b.SearchSpace.LearningRate[rand.Intn(len(b.SearchSpace.LearningRate))]
	}

	sum := 0.0
	for _, cfg := range bestConfigs {
		sum += cfg.Config.LearningRate
	}
	avg := sum / float64(len(bestConfigs))
	noise := (rand.Float64() - 0.5) * 0.1 * avg
	return math.Max(1e-9, avg+noise)
}

// interpolateBatchSize interpolates batch size from best configs
func (b *BayesianOptimizer) interpolateBatchSize(bestConfigs []TrialResult) int {
	if len(bestConfigs) == 0 {
		return b.SearchSpace.BatchSize[rand.Intn(len(b.SearchSpace.BatchSize))]
	}

	sum := 0
	for _, cfg := range bestConfigs {
		sum += cfg.Config.BatchSize
	}
	avg := float64(sum) / float64(len(bestConfigs))

	closest := b.SearchSpace.BatchSize[0]
	minDiff := math.Abs(avg - float64(closest))
	for _, size := range b.SearchSpace.BatchSize {
		diff := math.Abs(avg - float64(size))
		if diff < minDiff {
			minDiff = diff
			closest = size
		}
	}
	return closest
}

// mostCommonOptimizer returns the most common optimizer from best configs
func (b *BayesianOptimizer) mostCommonOptimizer(bestConfigs []TrialResult) string {
	if len(bestConfigs) == 0 {
		return b.SearchSpace.Optimizers[rand.Intn(len(b.SearchSpace.Optimizers))]
	}

	counts := make(map[string]int)
	for _, cfg := range bestConfigs {
		counts[cfg.Config.Optimizer]++
	}

	mostCommon := ""
	maxCount := 0
	for opt, cnt := range counts {
		if cnt > maxCount {
			maxCount = cnt
			mostCommon = opt
		}
	}
	if mostCommon == "" && len(b.SearchSpace.Optimizers) > 0 {
		return b.SearchSpace.Optimizers[0]
	}
	return mostCommon
}

// interpolateEpochs interpolates epochs from best configs
func (b *BayesianOptimizer) interpolateEpochs(bestConfigs []TrialResult) int {
	if len(bestConfigs) == 0 {
		return b.SearchSpace.Epochs[rand.Intn(len(b.SearchSpace.Epochs))]
	}

	sum := 0
	for _, cfg := range bestConfigs {
		sum += cfg.Config.Epochs
	}
	avg := float64(sum) / float64(len(bestConfigs))

	closest := b.SearchSpace.Epochs[0]
	minDiff := math.Abs(avg - float64(closest))
	for _, e := range b.SearchSpace.Epochs {
		diff := math.Abs(avg - float64(e))
		if diff < minDiff {
			minDiff = diff
			closest = e
		}
	}
	return closest
}

// AddResult adds a trial result to the optimizer
func (b *BayesianOptimizer) AddResult(result TrialResult) {
	b.Results = append(b.Results, result)
}
