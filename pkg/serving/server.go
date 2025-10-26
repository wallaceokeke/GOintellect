package serving

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gointellect/gointellect/pkg/core"
	"github.com/gointellect/gointellect/pkg/model"
	"github.com/gointellect/gointellect/pkg/nn"
)

// PredictionRequest represents a prediction request
type PredictionRequest struct {
	Input    [][]float64 `json:"input"`
	ModelID  string      `json:"model_id,omitempty"`
	BatchID  string      `json:"batch_id,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// PredictionResponse represents a prediction response
type PredictionResponse struct {
	Predictions [][]float64 `json:"predictions"`
	ModelID     string       `json:"model_id"`
	BatchID     string       `json:"batch_id"`
	Latency     int64        `json:"latency_ms"`
	Timestamp   time.Time    `json:"timestamp"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// ModelServer represents a model serving server
type ModelServer struct {
	Models      map[string]*nn.NeuralNetwork
	ModelManager *model.ModelManager
	Port        int
	Server      *http.Server
	mu          sync.RWMutex
}

// NewModelServer creates a new model server
func NewModelServer(port int, modelDir string) *ModelServer {
	return &ModelServer{
		Models:       make(map[string]*nn.NeuralNetwork),
		ModelManager: model.NewModelManager(modelDir),
		Port:         port,
	}
}

// Start starts the model server
func (s *ModelServer) Start() error {
	mux := http.NewServeMux()
	
	// Prediction endpoint
	mux.HandleFunc("/predict", s.handlePredict)
	
	// Batch prediction endpoint
	mux.HandleFunc("/predict/batch", s.handleBatchPredict)
	
	// Model management endpoints
	mux.HandleFunc("/models", s.handleListModels)
	mux.HandleFunc("/models/", s.handleModelOperations)
	
	// Health check endpoint
	mux.HandleFunc("/health", s.handleHealth)
	
	// Metrics endpoint
	mux.HandleFunc("/metrics", s.handleMetrics)
	
	s.Server = &http.Server{
		Addr:    fmt.Sprintf(":%d", s.Port),
		Handler: mux,
	}
	
	log.Printf("Starting model server on port %d", s.Port)
	return s.Server.ListenAndServe()
}

// Stop stops the model server
func (s *ModelServer) Stop(ctx context.Context) error {
	return s.Server.Shutdown(ctx)
}

// LoadModel loads a model into memory
func (s *ModelServer) LoadModel(modelID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	model, _, err := s.ModelManager.LoadModel(modelID)
	if err != nil {
		return err
	}
	
	s.Models[modelID] = model
	log.Printf("Loaded model: %s", modelID)
	return nil
}

// UnloadModel removes a model from memory
func (s *ModelServer) UnloadModel(modelID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	delete(s.Models, modelID)
	log.Printf("Unloaded model: %s", modelID)
}

// handlePredict handles single prediction requests
func (s *ModelServer) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	start := time.Now()
	
	var req PredictionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	// Use default model if not specified
	modelID := req.ModelID
	if modelID == "" {
		modelID = "default"
	}
	
	// Get model
	s.mu.RLock()
	model, exists := s.Models[modelID]
	s.mu.RUnlock()
	
	if !exists {
		http.Error(w, fmt.Sprintf("Model %s not found", modelID), http.StatusNotFound)
		return
	}
	
	// Convert input to tensor
	inputTensor := core.NewTensor2D(len(req.Input), len(req.Input[0]))
	for i, row := range req.Input {
		for j, val := range row {
			inputTensor.Set(val, i, j)
		}
	}
	
	// Make prediction
	predictions := model.Forward(inputTensor)
	
	// Convert predictions to response format
	predArray := make([][]float64, predictions.Shape[0])
	for i := 0; i < predictions.Shape[0]; i++ {
		predArray[i] = make([]float64, predictions.Shape[1])
		for j := 0; j < predictions.Shape[1]; j++ {
			predArray[i][j] = predictions.At(i, j)
		}
	}
	
	// Create response
	response := PredictionResponse{
		Predictions: predArray,
		ModelID:     modelID,
		BatchID:     req.BatchID,
		Latency:     time.Since(start).Milliseconds(),
		Timestamp:   time.Now(),
		Metadata:    req.Metadata,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleBatchPredict handles batch prediction requests
func (s *ModelServer) handleBatchPredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	start := time.Now()
	
	var req PredictionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	// Use default model if not specified
	modelID := req.ModelID
	if modelID == "" {
		modelID = "default"
	}
	
	// Get model
	s.mu.RLock()
	model, exists := s.Models[modelID]
	s.mu.RUnlock()
	
	if !exists {
		http.Error(w, fmt.Sprintf("Model %s not found", modelID), http.StatusNotFound)
		return
	}
	
	// Process batch
	batchSize := len(req.Input)
	if batchSize == 0 {
		http.Error(w, "Empty batch", http.StatusBadRequest)
		return
	}
	
	// Convert input to tensor
	inputTensor := core.NewTensor2D(batchSize, len(req.Input[0]))
	for i, row := range req.Input {
		for j, val := range row {
			inputTensor.Set(val, i, j)
		}
	}
	
	// Make predictions
	predictions := model.Forward(inputTensor)
	
	// Convert predictions to response format
	predArray := make([][]float64, predictions.Shape[0])
	for i := 0; i < predictions.Shape[0]; i++ {
		predArray[i] = make([]float64, predictions.Shape[1])
		for j := 0; j < predictions.Shape[1]; j++ {
			predArray[i][j] = predictions.At(i, j)
		}
	}
	
	// Create response
	response := PredictionResponse{
		Predictions: predArray,
		ModelID:     modelID,
		BatchID:     req.BatchID,
		Latency:     time.Since(start).Milliseconds(),
		Timestamp:   time.Now(),
		Metadata:    req.Metadata,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleListModels handles model listing requests
func (s *ModelServer) handleListModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	models, err := s.ModelManager.ListModels()
	if err != nil {
		http.Error(w, "Failed to list models", http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models)
}

// handleModelOperations handles model operations (load, unload, delete)
func (s *ModelServer) handleModelOperations(w http.ResponseWriter, r *http.Request) {
	// Extract model ID from URL path
	modelID := r.URL.Path[len("/models/"):]
	if modelID == "" {
		http.Error(w, "Model ID required", http.StatusBadRequest)
		return
	}
	
	switch r.Method {
	case http.MethodPost:
		// Load model
		if err := s.LoadModel(modelID); err != nil {
			http.Error(w, fmt.Sprintf("Failed to load model: %v", err), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "Model %s loaded successfully", modelID)
		
	case http.MethodDelete:
		// Unload model
		s.UnloadModel(modelID)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "Model %s unloaded successfully", modelID)
		
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleHealth handles health check requests
func (s *ModelServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"models":    len(s.Models),
		"uptime":    time.Since(time.Now()).String(), // This would be actual uptime in practice
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// handleMetrics handles metrics requests
func (s *ModelServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	metrics := map[string]interface{}{
		"models_loaded": len(s.Models),
		"server_port":    s.Port,
		"timestamp":      time.Now(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

// ModelClient provides a client for interacting with the model server
type ModelClient struct {
	BaseURL string
	Client  *http.Client
}

// NewModelClient creates a new model client
func NewModelClient(baseURL string) *ModelClient {
	return &ModelClient{
		BaseURL: baseURL,
		Client:  &http.Client{Timeout: 30 * time.Second},
	}
}

// Predict makes a prediction request
func (c *ModelClient) Predict(input [][]float64, modelID string) (*PredictionResponse, error) {
	req := PredictionRequest{
		Input:   input,
		ModelID: modelID,
	}
	
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	
	resp, err := c.Client.Post(c.BaseURL+"/predict", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("prediction failed with status: %d", resp.StatusCode)
	}
	
	var predictionResp PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&predictionResp); err != nil {
		return nil, err
	}
	
	return &predictionResp, nil
}

// BatchPredict makes a batch prediction request
func (c *ModelClient) BatchPredict(input [][]float64, modelID string) (*PredictionResponse, error) {
	req := PredictionRequest{
		Input:   input,
		ModelID: modelID,
	}
	
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	
	resp, err := c.Client.Post(c.BaseURL+"/predict/batch", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("batch prediction failed with status: %d", resp.StatusCode)
	}
	
	var predictionResp PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&predictionResp); err != nil {
		return nil, err
	}
	
	return &predictionResp, nil
}

// LoadModel loads a model on the server
func (c *ModelClient) LoadModel(modelID string) error {
	resp, err := c.Client.Post(c.BaseURL+"/models/"+modelID, "application/json", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to load model with status: %d", resp.StatusCode)
	}
	
	return nil
}

// UnloadModel unloads a model from the server
func (c *ModelClient) UnloadModel(modelID string) error {
	req, err := http.NewRequest("DELETE", c.BaseURL+"/models/"+modelID, nil)
	if err != nil {
		return err
	}
	
	resp, err := c.Client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to unload model with status: %d", resp.StatusCode)
	}
	
	return nil
}

// ListModels lists available models
func (c *ModelClient) ListModels() ([]model.ModelMetadata, error) {
	resp, err := c.Client.Get(c.BaseURL + "/models")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to list models with status: %d", resp.StatusCode)
	}
	
	var models []model.ModelMetadata
	if err := json.NewDecoder(resp.Body).Decode(&models); err != nil {
		return nil, err
	}
	
	return models, nil
}

// HealthCheck checks server health
func (c *ModelClient) HealthCheck() (map[string]interface{}, error) {
	resp, err := c.Client.Get(c.BaseURL + "/health")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}
	
	var health map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return nil, err
	}
	
	return health, nil
}
