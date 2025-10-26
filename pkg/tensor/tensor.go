package tensor

import "fmt"

// Tensor2D is a minimal 2D tensor backed by [][]float64
type Tensor2D struct {
    Rows int
    Cols int
    Data [][]float64
}

func NewTensor2D(rows, cols int) *Tensor2D {
    data := make([][]float64, rows)
    for i := range data {
        data[i] = make([]float64, cols)
    }
    return &Tensor2D{Rows: rows, Cols: cols, Data: data}
}

func From2DSlice(slice [][]float64) *Tensor2D {
    r := len(slice)
    c := 0
    if r > 0 {
        c = len(slice[0])
    }
    return &Tensor2D{Rows: r, Cols: c, Data: slice}
}

func (t *Tensor2D) Row(i int) []float64 {
    return t.Data[i]
}

func (t *Tensor2D) At(i, j int) float64 {
    return t.Data[i][j]
}

func (t *Tensor2D) Set(i, j int, v float64) {
    t.Data[i][j] = v
}

func (t *Tensor2D) String() string {
    s := ""
    for i := 0; i < t.Rows; i++ {
        s += fmt.Sprintf("%v\n", t.Data[i])
    }
    return s
}
