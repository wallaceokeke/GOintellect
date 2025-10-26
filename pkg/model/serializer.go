package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/gointellect/gointellect/pkg/nn"
)

// Alias Conv2DLayer to the type defined in the nn package to resolve undefined identifier.
type Conv2DLayer = nn.Conv2DLayer

// ModelMetadata contains metadata about a saved model
type ModelMetadata struct {
	Name         string                 `json:"name"`
	Version      string                 `json:"version"`
	CreatedAt    time.Time              `json:"created_at"`
	Description  string                 `json:"description"`
	Architecture string                 `json:"architecture"`
	Parameters   int                    `json:"parameters"`
	InputShape   []int                  `json:"input_shape"`
	OutputShape  []int                  `json:"output_shape"`
	Tags         []string               `json:"tags"`
	Metrics      map[string]float64     `json:"metrics"`
	Config       map[string]interface{} `json:"config"`
}

// Model represents a complete model with metadata
type Model struct {
	Metadata ModelMetadata `json:"metadata"`
	Layers   []LayerInfo   `json:"layers"`
	Weights  [][]float64   `json:"weights"`
	Biases   [][]float64   `json:"biases"`
}

// LayerInfo contains information about a layer
type LayerInfo struct {
	Type       string                 `json:"type"`
	InputSize  int                    `json:"input_size"`
	OutputSize int                    `json:"output_size"`
	Config     map[string]interface{} `json:"config"`
}

// ModelSerializer handles model serialization and deserialization
type ModelSerializer struct {
	Format string // "json", "binary", "protobuf"
}

// NewModelSerializer creates a new model serializer
func NewModelSerializer(format string) *ModelSerializer {
	return &ModelSerializer{Format: format}
}

// Save saves a neural network model to file
func (s *ModelSerializer) Save(model *nn.NeuralNetwork, filepath string, metadata ModelMetadata) error {
	switch s.Format {
	case "json":
		return s.saveJSON(model, filepath, metadata)
	case "binary":
		return s.saveBinary(model, filepath, metadata)
	default:
		return fmt.Errorf("unsupported format: %s", s.Format)
	}
}

// Load loads a neural network model from file
func (s *ModelSerializer) Load(filepath string) (*nn.NeuralNetwork, ModelMetadata, error) {
	switch s.Format {
	case "json":
		return s.loadJSON(filepath)
	case "binary":
		return s.loadBinary(filepath)
	default:
		return nil, ModelMetadata{}, fmt.Errorf("unsupported format: %s", s.Format)
	}
}

// saveJSON saves model in JSON format
func (s *ModelSerializer) saveJSON(model *nn.NeuralNetwork, filepath string, metadata ModelMetadata) error {
	// Extract layer information
	layers := make([]LayerInfo, len(model.Layers))
	for i, layer := range model.Layers {
		layers[i] = LayerInfo{
			Type:   s.getLayerType(layer),
			Config: s.getLayerConfig(layer),
		}
	}

	// Extract weights and biases
	params := model.Parameters()
	weights := make([][]float64, len(params))
	biases := make([][]float64, len(params))

	for i, param := range params {
		weights[i] = make([]float64, len(param.Data))
		copy(weights[i], param.Data)
	}

	// Create model structure
	modelData := Model{
		Metadata: metadata,
		Layers:   layers,
		Weights:  weights,
		Biases:   biases,
	}

	// Write to file
	data, err := json.MarshalIndent(modelData, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filepath, data, 0644)
}

// loadJSON loads model from JSON format
func (s *ModelSerializer) loadJSON(filepath string) (*nn.NeuralNetwork, ModelMetadata, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, ModelMetadata{}, err
	}

	var modelData Model
	if err := json.Unmarshal(data, &modelData); err != nil {
		return nil, ModelMetadata{}, err
	}

	// Reconstruct neural network
	layers := make([]nn.Layer, len(modelData.Layers))
	for i, layerInfo := range modelData.Layers {
		layer, err := s.createLayerFromInfo(layerInfo)
		if err != nil {
			return nil, ModelMetadata{}, err
		}
		layers[i] = layer
	}

	model := nn.NewNeuralNetwork(layers...)

	// Set parameters
	params := model.Parameters()
	for i, param := range params {
		if i < len(modelData.Weights) {
			copy(param.Data, modelData.Weights[i])
		}
	}

	return model, modelData.Metadata, nil
}

// saveBinary saves model in binary format (simplified)
func (s *ModelSerializer) saveBinary(model *nn.NeuralNetwork, filepath string, metadata ModelMetadata) error {
	// This is a simplified binary format
	// In practice, you'd use a more efficient format like Protocol Buffers

	// For now, we'll use JSON as a placeholder
	return s.saveJSON(model, filepath, metadata)
}

// loadBinary loads model from binary format
func (s *ModelSerializer) loadBinary(filepath string) (*nn.NeuralNetwork, ModelMetadata, error) {
	// Simplified binary loading
	return s.loadJSON(filepath)
}
// getLayerType returns the type of a layer
func (s *ModelSerializer) getLayerType(layer nn.Layer) string {
	switch layer.(type) {
	case *nn.DenseLayer:
		return "dense"
	case *nn.Conv2DLayer:
		return "conv2d"
	case *nn.MaxPool2DLayer:
		return "maxpool2d"
	case *nn.FlattenLayer:
		return "flatten"
	case *nn.DropoutLayer:
		return "dropout"
	case *nn.BatchNormLayer:
		return "batchnorm"
	case *nn.RNNLayer:
		return "rnn"
	case *nn.LSTMLayer:
		return "lstm"
	case *nn.GRULayer:
		return "gru"
	default:
		return "unknown"
	}
}

// getLayerConfig returns the configuration of a layer
func (s *ModelSerializer) getLayerConfig(layer nn.Layer) map[string]interface{} {
	config := make(map[string]interface{})
	
	switch l := layer.(type) {
	case *nn.DenseLayer:
		config["input_size"] = l.InputSize
		config["output_size"] = l.OutputSize
		config["activation"] = s.getActivationType(l.Activation)
	case *nn.Conv2DLayer:
		config["filters"] = l.Filters
		config["kernel_size"] = l.KernelSize
		config["stride"] = l.Stride
		config["padding"] = l.Padding
		config["activation"] = s.getActivationType(l.Activation)
	case *nn.MaxPool2DLayer:
		config["pool_size"] = l.PoolSize
		config["stride"] = l.Stride
		config["padding"] = l.Padding
	case *nn.DropoutLayer:
		config["rate"] = l.Rate
	case *nn.BatchNormLayer:
		config["momentum"] = l.Momentum
		config["epsilon"] = l.Epsilon
	case *nn.RNNLayer:
		config["input_size"] = l.InputSize
		config["hidden_size"] = l.HiddenSize
		config["activation"] = s.getActivationType(l.Activation)
	case *nn.LSTMLayer:
		config["input_size"] = l.InputSize
		config["hidden_size"] = l.HiddenSize
	case *nn.GRULayer:
		config["input_size"] = l.InputSize
		config["hidden_size"] = l.HiddenSize
	}
	
	return config
}

// getActivationType returns the type of an activation function
func (s *ModelSerializer) getActivationType(activation nn.ActivationFunction) string {
	if activation == nil {
		return "none"
	}

	switch activation.(type) {
	case *nn.ReLU:
		return "relu"
	case *nn.Sigmoid:
		return "sigmoid"
	case *nn.Tanh:
		return "tanh"
	case *nn.Softmax:
		return "softmax"
	default:
		return "unknown"
	}
}
// createLayerFromInfo creates a layer from layer information
func (s *ModelSerializer) createLayerFromInfo(info LayerInfo) (nn.Layer, error) {
	switch info.Type {
	case "dense":
		inputSize := int(info.Config["input_size"].(float64))
		outputSize := int(info.Config["output_size"].(float64))
		activation := s.createActivationFromType(info.Config["activation"].(string))
		return nn.NewDenseLayer(inputSize, outputSize, activation), nil
	case "conv2d":
		filters := int(info.Config["filters"].(float64))
		kernelSize := [2]int{
			int(info.Config["kernel_size"].([]interface{})[0].(float64)),
			int(info.Config["kernel_size"].([]interface{})[1].(float64)),
		}
		stride := [2]int{
			int(info.Config["stride"].([]interface{})[0].(float64)),
			int(info.Config["stride"].([]interface{})[1].(float64)),
		}
		padding := [2]int{
			int(info.Config["padding"].([]interface{})[0].(float64)),
			int(info.Config["padding"].([]interface{})[1].(float64)),
		}
		activation := s.createActivationFromType(info.Config["activation"].(string))
		return nn.NewConv2DLayer(filters, kernelSize, stride, padding, activation), nil
	case "maxpool2d":
		poolSize := [2]int{
			int(info.Config["pool_size"].([]interface{})[0].(float64)),
			int(info.Config["pool_size"].([]interface{})[1].(float64)),
		}
		stride := [2]int{
			int(info.Config["stride"].([]interface{})[0].(float64)),
			int(info.Config["stride"].([]interface{})[1].(float64)),
		}
		padding := [2]int{
			int(info.Config["padding"].([]interface{})[0].(float64)),
			int(info.Config["padding"].([]interface{})[1].(float64)),
		}
		return nn.NewMaxPool2DLayer(poolSize, stride, padding), nil
	case "flatten":
		return nn.NewFlattenLayer(), nil
	case "dropout":
		rate := info.Config["rate"].(float64)
		return nn.NewDropoutLayer(rate), nil
	case "batchnorm":
		// This would need the number of features, which isn't stored
		// For now, return a default
		return nn.NewBatchNormLayer(128), nil
	case "rnn":
		inputSize := int(info.Config["input_size"].(float64))
		hiddenSize := int(info.Config["hidden_size"].(float64))
		activation := s.createActivationFromType(info.Config["activation"].(string))
		return nn.NewRNNLayer(inputSize, hiddenSize, activation), nil
	case "lstm":
		inputSize := int(info.Config["input_size"].(float64))
		hiddenSize := int(info.Config["hidden_size"].(float64))
		return nn.NewLSTMLayer(inputSize, hiddenSize), nil
	case "gru":
		inputSize := int(info.Config["input_size"].(float64))
		hiddenSize := int(info.Config["hidden_size"].(float64))
		return nn.NewGRULayer(inputSize, hiddenSize), nil
	default:
		return nil, fmt.Errorf("unknown layer type: %s", info.Type)
	}
}

// createActivationFromType creates an activation function from type string
func (s *ModelSerializer) createActivationFromType(activationType string) nn.ActivationFunction {
	switch activationType {
	case "relu":
		return nn.NewReLU()
	case "sigmoid":
		return nn.NewSigmoid()
	case "tanh":
		return nn.NewTanh()
	case "softmax":
		return nn.NewSoftmax()
	default:
		return nil
	}
}

// ModelManager provides high-level model management
type ModelManager struct {
	Serializer *ModelSerializer
	ModelDir   string
}

// NewModelManager creates a new model manager
func NewModelManager(modelDir string) *ModelManager {
	return &ModelManager{
		Serializer: NewModelSerializer("json"),
		ModelDir:   modelDir,
	}
}

// SaveModel saves a model with automatic metadata generation
func (m *ModelManager) SaveModel(model *nn.NeuralNetwork, name string, description string) error {
	// Create model directory if it doesn't exist
	if err := os.MkdirAll(m.ModelDir, 0755); err != nil {
		return err
	}

	// Generate metadata
	metadata := ModelMetadata{
		Name:         name,
		Version:      "1.0.0",
		CreatedAt:    time.Now(),
		Description:  description,
		Architecture: "neural_network",
		Parameters:   m.countParameters(model),
		Tags:         []string{"gointellect", "neural_network"},
		Metrics:      make(map[string]float64),
		Config:       make(map[string]interface{}),
	}

	// Save model
	filepath := filepath.Join(m.ModelDir, name+".json")
	return m.Serializer.Save(model, filepath, metadata)
}

// LoadModel loads a model by name
func (m *ModelManager) LoadModel(name string) (*nn.NeuralNetwork, ModelMetadata, error) {
	filepath := filepath.Join(m.ModelDir, name+".json")
	return m.Serializer.Load(filepath)
}

// ListModels lists all available models
func (m *ModelManager) ListModels() ([]ModelMetadata, error) {
	files, err := filepath.Glob(filepath.Join(m.ModelDir, "*.json"))
	if err != nil {
		return nil, err
	}

	var models []ModelMetadata
	for _, file := range files {
		_, metadata, err := m.Serializer.Load(file)
		if err != nil {
			continue // Skip invalid files
		}
		models = append(models, metadata)
	}

	return models, nil
}

// DeleteModel deletes a model by name
func (m *ModelManager) DeleteModel(name string) error {
	filepath := filepath.Join(m.ModelDir, name+".json")
	return os.Remove(filepath)
}

// countParameters counts the total number of parameters in a model
func (m *ModelManager) countParameters(model *nn.NeuralNetwork) int {
	params := model.Parameters()
	total := 0
	for _, param := range params {
		total += param.Size()
	}
	return total
}

// ModelCheckpoint provides checkpointing functionality
type ModelCheckpoint struct {
	Manager       *ModelManager
	BestScore     float64
	LastEpoch     int
	CheckpointDir string
}

// NewModelCheckpoint creates a new model checkpoint
func NewModelCheckpoint(checkpointDir string) *ModelCheckpoint {
	return &ModelCheckpoint{
		Manager:       NewModelManager(checkpointDir),
		BestScore:     -math.Inf(1),
		LastEpoch:     0,
		CheckpointDir: checkpointDir,
	}
}

// SaveCheckpoint saves a checkpoint if the score is better
func (c *ModelCheckpoint) SaveCheckpoint(model *nn.NeuralNetwork, epoch int, score float64, metrics map[string]float64) error {
	if score > c.BestScore {
		c.BestScore = score
		c.LastEpoch = epoch

		// Save best model
		metadata := ModelMetadata{
			Name:         "best_model",
			Version:      "1.0.0",
			CreatedAt:    time.Now(),
			Description:  fmt.Sprintf("Best model from epoch %d with score %.4f", epoch, score),
			Architecture: "neural_network",
			Parameters:   c.Manager.countParameters(model),
			Tags:         []string{"checkpoint", "best"},
			Metrics:      metrics,
			Config:       make(map[string]interface{}),
		}

		filepath := filepath.Join(c.CheckpointDir, "best_model.json")
		return c.Manager.Serializer.Save(model, filepath, metadata)
	}

	// Save regular checkpoint
	metadata := ModelMetadata{
		Name:         fmt.Sprintf("checkpoint_epoch_%d", epoch),
		Version:      "1.0.0",
		CreatedAt:    time.Now(),
		Description:  fmt.Sprintf("Checkpoint from epoch %d with score %.4f", epoch, score),
		Architecture: "neural_network",
		Parameters:   c.Manager.countParameters(model),
		Tags:         []string{"checkpoint"},
		Metrics:      metrics,
		Config:       make(map[string]interface{}),
	}

	filepath := filepath.Join(c.CheckpointDir, fmt.Sprintf("checkpoint_epoch_%d.json", epoch))
	return c.Manager.Serializer.Save(model, filepath, metadata)
}

// LoadBestModel loads the best saved model
func (c *ModelCheckpoint) LoadBestModel() (*nn.NeuralNetwork, ModelMetadata, error) {
	return c.Manager.LoadModel("best_model")
}
