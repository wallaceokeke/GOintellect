package main

import (
    "fmt"
    "log"
    "os"
    "strconv"

    "github.com/gointellect/gointellect/pkg/learn"
)

func printHelp() {
    fmt.Println("gointellect - simple AI utilities")
    fmt.Println()
    fmt.Println("Usage:")
    fmt.Println("  gointellect train linear x1,x2,... y1,y2,...")
    fmt.Println("  gointellect predict linear m b x")
    fmt.Println("  gointellect train perceptron n lr epochs  (simple demo with synthetic data)")
    fmt.Println("  gointellect help")
}

func main() {
    if len(os.Args) < 2 {
        printHelp()
        return
    }
    cmd := os.Args[1]
    switch cmd {
    case "help":
        printHelp()
    case "train":
        if len(os.Args) < 3 {
            printHelp(); return
        }
        model := os.Args[2]
        switch model {
        case "linear":
            if len(os.Args) < 5 {
                fmt.Println("train linear requires X and Y CSV lists"); return
            }
            xs := parseFloatList(os.Args[3])
            ys := parseFloatList(os.Args[4])
            m, b, err := learn.LinearRegression1D(xs, ys)
            if err != nil {
                log.Fatal(err)
            }
            fmt.Printf("Trained Linear Regression -> m: %f b: %f\n", m, b)
        case "perceptron":
            if len(os.Args) < 6 {
                fmt.Println("train perceptron requires n lr epochs"); return
            }
            n, _ := strconv.Atoi(os.Args[3])
            lr, _ := strconv.ParseFloat(os.Args[4], 64)
            epochs, _ := strconv.Atoi(os.Args[5])
            p := learn.NewPerceptron(n, lr)
            // small synthetic dataset: AND function for n==2
            X := [][]float64{{0,0},{0,1},{1,0},{1,1}}
            Y := []int{0,0,0,1}
            p.Train(X, Y, epochs)
            fmt.Printf("Trained Perceptron Weights: %v Bias: %f\n", p.Weights, p.Bias)
        default:
            fmt.Println("unknown model:", model)
        }
    case "predict":
        if len(os.Args) < 3 {
            printHelp(); return
        }
        model := os.Args[2]
        switch model {
        case "linear":
            if len(os.Args) < 6 {
                fmt.Println("predict linear requires m b x"); return
            }
            m, _ := strconv.ParseFloat(os.Args[3], 64)
            b, _ := strconv.ParseFloat(os.Args[4], 64)
            x, _ := strconv.ParseFloat(os.Args[5], 64)
            y := m*x + b
            fmt.Printf("Prediction: %f\n", y)
        default:
            fmt.Println("unknown model:", model)
        }
    default:
        printHelp()
    }
}

func parseFloatList(s string) []float64 {
    parts := []rune(s)
    // simple comma split without strings package to keep minimal
    var cur string
    var out []float64
    for _, r := range parts {
        if r == ',' {
            if cur != "" {
                v, _ := strconv.ParseFloat(cur, 64)
                out = append(out, v)
                cur = ""
            }
        } else {
            cur += string(r)
        }
    }
    if cur != "" {
        v, _ := strconv.ParseFloat(cur, 64)
        out = append(out, v)
    }
    return out
}
