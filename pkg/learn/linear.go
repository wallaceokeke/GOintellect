package learn

import (
	"errors"
	"fmt"
	"math"
	"time"
)

// RegressionConfig holds configuration for regression models
type RegressionConfig struct {
	ConfidenceLevel float64
	CrossValidation bool
	CVFolds         int
}

// RegressionResult represents comprehensive regression output
type RegressionResult struct {
	Slope        float64
	Intercept    float64
	RSquared     float64
	RMSE         float64
	MAE          float64
	TrainingTime time.Duration
	Residuals    []float64
	Predictions  []float64
}

// LinearRegression1D implements advanced linear regression
func LinearRegression1D(X, y []float64) (float64, float64, *RegressionResult, error) {
	return LinearRegression1DWithConfig(X, y, nil)
}

// LinearRegression1DWithConfig performs regression with configuration
func LinearRegression1DWithConfig(X, y []float64, config *RegressionConfig) (float64, float64, *RegressionResult, error) {
	startTime := time.Now()
	
	// Input validation
	if len(X) != len(y) || len(X) == 0 {
		return 0, 0, nil, errors.New("input length mismatch or zero")
	}
	
	n := float64(len(X))
	var sumX, sumY, sumXY, sumX2 float64
	
	for i := range X {
		sumX += X[i]
		sumY += y[i]
		sumXY += X[i] * y[i]
		sumX2 += X[i] * X[i]
	}
	
	denom := n*sumX2 - sumX*sumX
	if denom == 0 {
		return 0, 0, nil, errors.New("singular matrix")
	}
	
	slope := (n*sumXY - sumX*sumY) / denom
	intercept := (sumY - slope*sumX) / n
	
	// Calculate comprehensive results
	result := calculateMetrics(X, y, slope, intercept)
	result.TrainingTime = time.Since(startTime)
	
	return slope, intercept, result, nil
}

func calculateMetrics(X, y []float64, slope, intercept float64) *RegressionResult {
	n := float64(len(X))
	result := &RegressionResult{
		Slope:     slope,
		Intercept: intercept,
		Residuals: make([]float64, len(X)),
		Predictions: make([]float64, len(X)),
	}
	
	ssResidual, ssTotal := 0.0, 0.0
	meanY := mean(y)
	
	for i := range X {
		prediction := slope*X[i] + intercept
		result.Predictions[i] = prediction
		residual := y[i] - prediction
		result.Residuals[i] = residual
		ssResidual += residual * residual
		ssTotal += (y[i] - meanY) * (y[i] - meanY)
	}
	
	result.RSquared = 1 - (ssResidual / ssTotal)
	result.RMSE = math.Sqrt(ssResidual / n)
	result.MAE = meanAbsoluteError(result.Residuals)
	
	return result
}

func mean(data []float64) float64 {
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func meanAbsoluteError(residuals []float64) float64 {
	sum := 0.0
	for _, r := range residuals {
		sum += math.Abs(r)
	}
	return sum / float64(len(residuals))
}
