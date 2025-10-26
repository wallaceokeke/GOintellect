package main

import (
    "fmt"
    "github.com/gointellect/gointellect/pkg/learn"
)

func main() {
    // Linear regression example: y = 2x + 1
    X := []float64{1,2,3,4,5}
    Y := []float64{3,5,7,9,11}
    m, b, _ := learn.LinearRegression1D(X, Y)
    fmt.Printf("Learned linear model -> m: %f b: %f\n", m, b)
    fmt.Printf("Predict for x=10 -> %f\n", m*10+b)

    // Perceptron demo
    p := learn.NewPerceptron(2, 0.1)
    Xp := [][]float64{{0,0},{0,1},{1,0},{1,1}}
    Yp := []int{0,0,0,1}
    p.Train(Xp, Yp, 10)
    fmt.Printf("Perceptron weights: %v bias: %f\n", p.Weights, p.Bias)
}
