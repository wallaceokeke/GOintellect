package learn

import (
	"fmt"
	"math"
)

// ExampleAdvancedRegression demonstrates the ultimate regression capabilities
func ExampleAdvancedRegression() {
	// Sample data: House prices vs. square footage
	X := []float64{1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000}
	y := []float64{300000, 400000, 500000, 550000, 600000, 650000, 700000, 750000, 800000}
	
	// Configure advanced regression
	config := &RegressionConfig{
		Type:            LinearRegression1D,
		ConfidenceLevel: 0.95,
		CrossValidation: true,
		CVFolds:         3,
	}
	
	// Perform comprehensive regression
	slope, intercept, result, err := LinearRegression1DWithConfig(X, y, config)
	if err != nil {
		fmt.Printf("Regression failed: %v\n", err)
		return
	}
	
	// Display comprehensive results
	fmt.Println("🏠 ADVANCED HOUSE PRICE REGRESSION ANALYSIS")
	fmt.Println("===========================================")
	fmt.Printf("Model: Price = %.2f * SqFt + %.2f\n", slope, intercept)
	fmt.Println(result.Summary())
	
	// Make prediction with confidence
	prediction := slope*6000 + intercept
	fmt.Printf("\n🔮 Prediction for 6000 sqft: $%.2f\n", prediction)
	
	// Show outlier analysis
	if len(result.Outliers) > 0 {
		fmt.Printf("\n⚠️  Detected %d outliers at indices: %v\n", 
			len(result.Outliers), result.Outliers)
	}
	
	// Show most influential points
	maxInfluence := 0.0
	influentialPoint := -1
	for i, cookD := range result.CookDistance {
		if cookD > maxInfluence {
			maxInfluence = cookD
			influentialPoint = i
		}
	}
	if influentialPoint >= 0 && maxInfluence > 0.5 {
		fmt.Printf("📊 Most influential point: index %d (Cook's D: %.4f)\n", 
			influentialPoint, maxInfluence)
	}
}

// ExampleRobustRegression shows robust regression with outliers
func ExampleRobustRegression() {
	// Data with outliers
	X := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100} // Outlier at x=100
	y := []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 500} // Corresponding outlier
	
	config := &RegressionConfig{
		Type:            RobustRegression,
		RobustMethod:    "huber",
		ConfidenceLevel: 0.95,
	}
	
	slope, intercept, result, err := LinearRegression1DWithConfig(X, y, config)
	if err != nil {
		fmt.Printf("Robust regression failed: %v\n", err)
		return
	}
	
	fmt.Println("🛡️ ROBUST REGRESSION (Outlier Resistant)")
	fmt.Println("========================================")
	fmt.Printf("Model: y = %.4f * x + %.4f\n", slope, intercept)
	fmt.Printf("R²: %.4f | RMSE: %.4f\n", result.RSquared, result.RMSE)
	fmt.Printf("Detected %d outliers\n", len(result.Outliers))
}

// GenerateSampleData creates synthetic data for testing
func GenerateSampleData(n int, slope, intercept, noiseStdDev float64) ([]float64, []float64) {
	X := make([]float64, n)
	y := make([]float64, n)
	
	for i := 0; i < n; i++ {
		X[i] = float64(i) + 1
		noise := (rand.Float64() - 0.5) * 2 * noiseStdDev
		y[i] = slope*X[i] + intercept + noise
	}
	
	return X, y
}
