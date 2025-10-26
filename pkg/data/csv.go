package data

import (
    "encoding/csv"
    "io"
    "os"
    "strconv"
)

// LoadCSV loads a CSV with numeric values (no header) into [][]float64
func LoadCSV(path string) ([][]float64, error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer f.Close()
    r := csv.NewReader(f)
    var out [][]float64
    for {
        rec, err := r.Read()
        if err == io.EOF {
            break
        }
        if err != nil {
            return nil, err
        }
        row := make([]float64, len(rec))
        for i, s := range rec {
            v, err := strconv.ParseFloat(s, 64)
            if err != nil {
                return nil, err
            }
            row[i] = v
        }
        out = append(out, row)
    }
    return out, nil
}
