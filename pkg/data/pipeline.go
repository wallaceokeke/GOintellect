package data

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/wallaceokeke/GOintellect/pkg/core"
)

// Dataset represents a dataset with features and targets
type Dataset struct {
	Features *core.Tensor
	Targets  *core.Tensor
	FeatureNames []string
	TargetNames  []string
}

// NewDataset creates a new dataset
func NewDataset(features, targets *core.Tensor) *Dataset {
	return &Dataset{
		Features: features,
		Targets:  targets,
	}
}

// Loader interface for loading data from different sources
type Loader interface {
	Load(path string) (*Dataset, error)
}

// CSVLoader loads data from CSV files
type CSVLoader struct {
	HasHeader     bool
	FeatureCols   []int
	TargetCols    []int
	Delimiter     rune
	SkipRows      int
}

// NewCSVLoader creates a new CSV loader
func NewCSVLoader() *CSVLoader {
	return &CSVLoader{
		HasHeader:   true,
		Delimiter:   ',',
		SkipRows:    0,
	}
}

// SetFeatureColumns sets which columns to use as features
func (c *CSVLoader) SetFeatureColumns(cols []int) *CSVLoader {
	c.FeatureCols = cols
	return c
}

// SetTargetColumns sets which columns to use as targets
func (c *CSVLoader) SetTargetColumns(cols []int) *CSVLoader {
	c.TargetCols = cols
	return c
}

// SetDelimiter sets the CSV delimiter
func (c *CSVLoader) SetDelimiter(delimiter rune) *CSVLoader {
	c.Delimiter = delimiter
	return c
}

// SetSkipRows sets number of rows to skip
func (c *CSVLoader) SetSkipRows(rows int) *CSVLoader {
	c.SkipRows = rows
	return c
}

// Load loads data from CSV file
func (c *CSVLoader) Load(path string) (*Dataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = c.Delimiter

	var rows [][]string
	var headers []string

	// Read all rows
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		rows = append(rows, record)
	}

	if len(rows) == 0 {
		return nil, fmt.Errorf("no data found in file")
	}

	// Handle headers
	startRow := 0
	if c.HasHeader {
		headers = rows[0]
		startRow = 1
	}

	// Skip specified rows
	startRow += c.SkipRows

	// Determine feature and target columns
	featureCols := c.FeatureCols
	targetCols := c.TargetCols

	if len(featureCols) == 0 {
		// Use all columns except targets as features
		allCols := make([]int, len(rows[startRow]))
		for i := range allCols {
			allCols[i] = i
		}
		featureCols = allCols
		for _, targetCol := range targetCols {
			for i, col := range featureCols {
				if col == targetCol {
					featureCols = append(featureCols[:i], featureCols[i+1:]...)
					break
				}
			}
		}
	}

	if len(targetCols) == 0 {
		// Use last column as target
		targetCols = []int{len(rows[startRow]) - 1}
	}

	// Convert to numeric data
	numRows := len(rows) - startRow
	numFeatures := len(featureCols)
	numTargets := len(targetCols)

	features := core.NewTensor2D(numRows, numFeatures)
	targets := core.NewTensor2D(numRows, numTargets)

	for i, row := range rows[startRow:] {
		// Parse features
		for j, colIdx := range featureCols {
			if colIdx >= len(row) {
				return nil, fmt.Errorf("column index %d out of range", colIdx)
			}
			val, err := strconv.ParseFloat(strings.TrimSpace(row[colIdx]), 64)
			if err != nil {
				return nil, fmt.Errorf("cannot parse feature value '%s' at row %d, col %d: %v", row[colIdx], i, colIdx, err)
			}
			features.Set(val, i, j)
		}

		// Parse targets
		for j, colIdx := range targetCols {
			if colIdx >= len(row) {
				return nil, fmt.Errorf("column index %d out of range", colIdx)
			}
			val, err := strconv.ParseFloat(strings.TrimSpace(row[colIdx]), 64)
			if err != nil {
				return nil, fmt.Errorf("cannot parse target value '%s' at row %d, col %d: %v", row[colIdx], i, colIdx, err)
			}
			targets.Set(val, i, j)
		}
	}

	// Extract column names
	var featureNames []string
	var targetNames []string

	if c.HasHeader && len(headers) > 0 {
		for _, colIdx := range featureCols {
			if colIdx < len(headers) {
				featureNames = append(featureNames, headers[colIdx])
			}
		}
		for _, colIdx := range targetCols {
			if colIdx < len(headers) {
				targetNames = append(targetNames, headers[colIdx])
			}
		}
	}

	return &Dataset{
		Features:     features,
		Targets:      targets,
		FeatureNames: featureNames,
		TargetNames:  targetNames,
	}, nil
}

// Transformer interface for data transformations
type Transformer interface {
	Fit(dataset *Dataset) error
	Transform(dataset *Dataset) (*Dataset, error)
	FitTransform(dataset *Dataset) (*Dataset, error)
}

// StandardScaler standardizes features by removing mean and scaling to unit variance
type StandardScaler struct {
	Means   []float64
	Stds    []float64
	Fitted  bool
}

// NewStandardScaler creates a new standard scaler
func NewStandardScaler() *StandardScaler {
	return &StandardScaler{
		Fitted: false,
	}
}

// Fit computes the mean and std for later scaling
func (s *StandardScaler) Fit(dataset *Dataset) error {
	features := dataset.Features
	numFeatures := features.Shape[1]
	
	s.Means = make([]float64, numFeatures)
	s.Stds = make([]float64, numFeatures)
	
	// Compute means
	for j := 0; j < numFeatures; j++ {
		sum := 0.0
		for i := 0; i < features.Shape[0]; i++ {
			sum += features.At(i, j)
		}
		s.Means[j] = sum / float64(features.Shape[0])
	}
	
	// Compute standard deviations
	for j := 0; j < numFeatures; j++ {
		sumSq := 0.0
		for i := 0; i < features.Shape[0]; i++ {
			diff := features.At(i, j) - s.Means[j]
			sumSq += diff * diff
		}
		s.Stds[j] = math.Sqrt(sumSq / float64(features.Shape[0]))
	}
	
	s.Fitted = true
	return nil
}

// Transform performs standardization
func (s *StandardScaler) Transform(dataset *Dataset) (*Dataset, error) {
	if !s.Fitted {
		return nil, fmt.Errorf("scaler must be fitted before transform")
	}
	
	features := dataset.Features
	numSamples := features.Shape[0]
	numFeatures := features.Shape[1]
	
	scaledFeatures := core.NewTensor2D(numSamples, numFeatures)
	
	for i := 0; i < numSamples; i++ {
		for j := 0; j < numFeatures; j++ {
			if s.Stds[j] == 0 {
				scaledFeatures.Set(0, i, j)
			} else {
				scaled := (features.At(i, j) - s.Means[j]) / s.Stds[j]
				scaledFeatures.Set(scaled, i, j)
			}
		}
	}
	
	return &Dataset{
		Features:     scaledFeatures,
		Targets:      dataset.Targets,
		FeatureNames: dataset.FeatureNames,
		TargetNames:  dataset.TargetNames,
	}, nil
}

// FitTransform fits the scaler and transforms the data
func (s *StandardScaler) FitTransform(dataset *Dataset) (*Dataset, error) {
	if err := s.Fit(dataset); err != nil {
		return nil, err
	}
	return s.Transform(dataset)
}

// MinMaxScaler scales features to a given range
type MinMaxScaler struct {
	Min     []float64
	Max     []float64
	Range   [2]float64
	Fitted  bool
}

// NewMinMaxScaler creates a new min-max scaler
func NewMinMaxScaler(min, max float64) *MinMaxScaler {
	return &MinMaxScaler{
		Range:  [2]float64{min, max},
		Fitted: false,
	}
}

// Fit computes the min and max for later scaling
func (m *MinMaxScaler) Fit(dataset *Dataset) error {
	features := dataset.Features
	numFeatures := features.Shape[1]
	
	m.Min = make([]float64, numFeatures)
	m.Max = make([]float64, numFeatures)
	
	// Initialize with first row
	for j := 0; j < numFeatures; j++ {
		m.Min[j] = features.At(0, j)
		m.Max[j] = features.At(0, j)
	}
	
	// Find min and max
	for i := 1; i < features.Shape[0]; i++ {
		for j := 0; j < numFeatures; j++ {
			val := features.At(i, j)
			if val < m.Min[j] {
				m.Min[j] = val
			}
			if val > m.Max[j] {
				m.Max[j] = val
			}
		}
	}
	
	m.Fitted = true
	return nil
}

// Transform performs min-max scaling
func (m *MinMaxScaler) Transform(dataset *Dataset) (*Dataset, error) {
	if !m.Fitted {
		return nil, fmt.Errorf("scaler must be fitted before transform")
	}
	
	features := dataset.Features
	numSamples := features.Shape[0]
	numFeatures := features.Shape[1]
	
	scaledFeatures := core.NewTensor2D(numSamples, numFeatures)
	
	for i := 0; i < numSamples; i++ {
		for j := 0; j < numFeatures; j++ {
			if m.Max[j] == m.Min[j] {
				scaledFeatures.Set(m.Range[0], i, j)
			} else {
				scaled := (features.At(i, j)-m.Min[j])/(m.Max[j]-m.Min[j])*(m.Range[1]-m.Range[0]) + m.Range[0]
				scaledFeatures.Set(scaled, i, j)
			}
		}
	}
	
	return &Dataset{
		Features:     scaledFeatures,
		Targets:      dataset.Targets,
		FeatureNames: dataset.FeatureNames,
		TargetNames:  dataset.TargetNames,
	}, nil
}

// FitTransform fits the scaler and transforms the data
func (m *MinMaxScaler) FitTransform(dataset *Dataset) (*Dataset, error) {
	if err := m.Fit(dataset); err != nil {
		return nil, err
	}
	return m.Transform(dataset)
}

// LabelEncoder encodes categorical labels as integers
type LabelEncoder struct {
	Labels   map[string]int
	Classes  []string
	Fitted   bool
}

// NewLabelEncoder creates a new label encoder
func NewLabelEncoder() *LabelEncoder {
	return &LabelEncoder{
		Labels:  make(map[string]int),
		Fitted:  false,
	}
}

// Fit learns the unique labels
func (l *LabelEncoder) Fit(dataset *Dataset) error {
	// For simplicity, assume targets are in string format
	// In practice, you'd need to handle different data types
	l.Labels = make(map[string]int)
	l.Classes = []string{}
	
	// This is a simplified implementation
	// In practice, you'd need to handle string targets properly
	l.Fitted = true
	return nil
}

// OneHotEncoder encodes categorical features as one-hot vectors
type OneHotEncoder struct {
	Categories [][]string
	Fitted     bool
}

// NewOneHotEncoder creates a new one-hot encoder
func NewOneHotEncoder() *OneHotEncoder {
	return &OneHotEncoder{
		Fitted: false,
	}
}

// Pipeline chains multiple transformers
type Pipeline struct {
	Steps []Transformer
}

// NewPipeline creates a new pipeline
func NewPipeline(steps ...Transformer) *Pipeline {
	return &Pipeline{
		Steps: steps,
	}
}

// Fit fits all transformers in the pipeline
func (p *Pipeline) Fit(dataset *Dataset) error {
	currentDataset := dataset
	for _, step := range p.Steps {
		if err := step.Fit(currentDataset); err != nil {
			return err
		}
	}
	return nil
}

// Transform applies all transformers in the pipeline
func (p *Pipeline) Transform(dataset *Dataset) (*Dataset, error) {
	currentDataset := dataset
	for _, step := range p.Steps {
		transformed, err := step.Transform(currentDataset)
		if err != nil {
			return nil, err
		}
		currentDataset = transformed
	}
	return currentDataset, nil
}

// FitTransform fits and transforms the data
func (p *Pipeline) FitTransform(dataset *Dataset) (*Dataset, error) {
	currentDataset := dataset
	for _, step := range p.Steps {
		transformed, err := step.FitTransform(currentDataset)
		if err != nil {
			return nil, err
		}
		currentDataset = transformed
	}
	return currentDataset, nil
}

// Splitter interface for splitting datasets
type Splitter interface {
	Split(dataset *Dataset) ([]*Dataset, error)
}

// TrainTestSplitter splits data into training and testing sets
type TrainTestSplitter struct {
	TestSize    float64
	RandomState int
	Shuffle     bool
}

// NewTrainTestSplitter creates a new train-test splitter
func NewTrainTestSplitter(testSize float64) *TrainTestSplitter {
	return &TrainTestSplitter{
		TestSize:    testSize,
		RandomState: 42,
		Shuffle:     true,
	}
}

// Split splits the dataset into training and testing sets
func (s *TrainTestSplitter) Split(dataset *Dataset) ([]*Dataset, error) {
	numSamples := dataset.Features.Shape[0]
	numTestSamples := int(float64(numSamples) * s.TestSize)
	numTrainSamples := numSamples - numTestSamples
	
	// Create indices
	indices := make([]int, numSamples)
	for i := range indices {
		indices[i] = i
	}
	
	// Shuffle if requested
	if s.Shuffle {
		// Initialize random seed
		rand.Seed(time.Now().UnixNano())
		// Fisher-Yates shuffle
		for i := len(indices) - 1; i > 0; i-- {
			j := rand.Intn(i + 1)
			indices[i], indices[j] = indices[j], indices[i]
		}
	}
	
	// Split features
	trainFeatures := core.NewTensor2D(numTrainSamples, dataset.Features.Shape[1])
	testFeatures := core.NewTensor2D(numTestSamples, dataset.Features.Shape[1])
	
	for i := 0; i < numTrainSamples; i++ {
		idx := indices[i]
		for j := 0; j < dataset.Features.Shape[1]; j++ {
			trainFeatures.Set(dataset.Features.At(idx, j), i, j)
		}
	}
	
	for i := 0; i < numTestSamples; i++ {
		idx := indices[numTrainSamples+i]
		for j := 0; j < dataset.Features.Shape[1]; j++ {
			testFeatures.Set(dataset.Features.At(idx, j), i, j)
		}
	}
	
	// Split targets
	trainTargets := core.NewTensor2D(numTrainSamples, dataset.Targets.Shape[1])
	testTargets := core.NewTensor2D(numTestSamples, dataset.Targets.Shape[1])
	
	for i := 0; i < numTrainSamples; i++ {
		idx := indices[i]
		for j := 0; j < dataset.Targets.Shape[1]; j++ {
			trainTargets.Set(dataset.Targets.At(idx, j), i, j)
		}
	}
	
	for i := 0; i < numTestSamples; i++ {
		idx := indices[numTrainSamples+i]
		for j := 0; j < dataset.Targets.Shape[1]; j++ {
			testTargets.Set(dataset.Targets.At(idx, j), i, j)
		}
	}
	
	trainDataset := &Dataset{
		Features:     trainFeatures,
		Targets:      trainTargets,
		FeatureNames: dataset.FeatureNames,
		TargetNames:  dataset.TargetNames,
	}
	
	testDataset := &Dataset{
		Features:     testFeatures,
		Targets:      testTargets,
		FeatureNames: dataset.FeatureNames,
		TargetNames:  dataset.TargetNames,
	}
	
	return []*Dataset{trainDataset, testDataset}, nil
}
