package learn

import "errors"

// LinearRegressionSimple implements closed-form linear regression for single-feature X.
// X is slice of float64 inputs, y is targets.
// Returns slope (m) and intercept (b) for y = m*x + b
func LinearRegression1D(X, y []float64) (float64, float64, error) {
    if len(X) != len(y) || len(X) == 0 {
        return 0, 0, errors.New("input length mismatch or zero")
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
        return 0, 0, errors.New("singular matrix")
    }
    m := (n*sumXY - sumX*sumY) / denom
    b := (sumY - m*sumX) / n
    return m, b, nil
}
