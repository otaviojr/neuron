use super::{MatrixMathExecutor, Tensor};

#[derive(Copy, Clone)]
pub struct MatrixMathCPU;

impl MatrixMathCPU {
  pub fn init() -> Self {
    MatrixMathCPU
  }
}

impl MatrixMathExecutor for MatrixMathCPU {

  fn add(&self, a: &mut Tensor, b: &Tensor) -> Tensor {
      // Check that the tensors are the same size
      assert!(a.rows == b.rows && a.cols == b.cols);

      // Perform element-wise addition
      for i in 0..a.data.len() {
          a.data[i] = a.data[i] + b.data[i];
      }

      a.clone()
  }

  fn sub(&self, a: &mut Tensor, b: &Tensor) -> Tensor {
      // Check that the tensors are the same size
      assert!(a.rows == b.rows && a.cols == b.cols);

      // Perform element-wise subtraction
      for i in 0..a.data.len() {
          a.data[i] = a.data[i] - b.data[i];
      }

      a.clone()
  }

  fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
    // Check that the tensors are compatible for multiplication
    //if a.cols != b.rows {
    //  return Tensor::zeros(0,0);
    //}
    assert!(a.cols == b.rows);

    // Create a new tensor to store the result
    let mut result = Tensor::new(a.rows, b.cols);

    // Perform matrix multiplication
    for i in 0..a.rows {
      for j in 0..b.cols {
        let mut sum = 0.0;
        for k in 0..a.cols {
          sum += a.get(i, k) * b.get(k, j);
        }
        result.set(i, j, sum);
      }
    }

    result
  }

  fn mul_wise(&self, a: &mut Tensor, b: &Tensor) -> Tensor {
    // Check that the tensors are compatible for multiplication
    assert!(a.rows == b.rows && a.cols == b.cols);

    let data = a.mut_data();
    for (v1, v2) in data.iter_mut().zip(b.data().iter()) {
      *v1 *= v2;
    }

    a.clone()
  }

  fn div(&self, a: &mut Tensor, b: &Tensor) -> Tensor {
      // Check that the tensors are the same size
      assert!(a.rows == b.rows && a.cols == b.cols);

      // Perform element-wise division
      for i in 0..a.data.len() {
          a.data[i] = a.data[i] / b.data[i];
      }

      a.clone()
  }

  fn dot(&self, a: &Tensor, b: &Tensor) -> f32 {
    let mut result = 0.0;

    // Check that the tensors are compatible for the dot product
    assert!(a.rows == b.rows && a.cols == b.cols);

    // Calculate the dot product
    for i in 0..a.rows {
        for j in 0..a.cols {
            result += a.get(i, j) * b.get(i, j);
        }
    }

    result
  }

  fn transpose(&self, a: &mut Tensor) -> Tensor {
    // Transpose the matrix
    for i in 0..a.rows {
        for j in 0..a.cols {
            if i < j {
                let temp = a.get(i, j);
                a.set(i, j, a.get(j, i));
                a.set(j, i, temp);
            }
        }
    }

    a.reshape(a.cols, a.rows).clone()
  }

  fn add_value(&self, a: &mut Tensor, value: f32) -> Tensor {
    for i in 0..a.rows {
      for j in 0..a.cols {
        let val = a.get(i, j);
        a.set(i, j, val + value);
      }
    }

    a.clone()
  }

  fn div_value(&self, a: &mut Tensor, value: f32) -> Tensor {
    // Create a new tensor to store the result
    for i in 0..a.rows {
      for j in 0..a.cols {
        let val = a.get(i, j);
        a.set(i, j, val/value);
      }
    }

    a.clone()
  }

  fn mul_value(&self, a: &mut Tensor, value: f32) -> Tensor {
    // Create a new tensor to store the result
    for i in 0..a.rows {
      for j in 0..a.cols {
        let val = a.get(i, j);
        a.set(i, j, val*value);
      }
    }
    a.clone()
  }

  fn sum_row(&self, a:&Tensor) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::new(a.rows, 1);

    for i in 0..a.rows {
      let mut sum: f32 = 0.0;
      for j in 0..a.cols {
        sum += a.get(i, j);
      }
      result.set(i, 0, sum);
    }

    result
  }

  fn broadcast(&self, a: &Tensor, rows: usize, cols: usize) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::new(rows, cols);

    if a.rows == rows {
      for i in 0..a.rows {
        for j in 0..result.cols {
          result.set(i, j, a.get(i, 0));
        }
      }  
    } else if a.cols == cols {
      for i in 0..result.rows {
        for j in 0..a.cols {
          result.set(i, j, a.get(0, j));
        }
      }  
    }

    result
  }

  fn pad(&self, a: &Tensor, pad_row: usize, pad_col: usize) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::new(a.rows() + pad_row*2, a.cols() + pad_col*2);

    for i in 0..a.rows() {
      for j in 0..a.cols() {
          result.set(i+pad_row, j+pad_col,a.get(i,j));
      }
    }

    result
  }

  fn zero(&self, a: &mut Tensor) -> Tensor {
    // Create a new tensor to store the result
    for i in 0..a.rows {
      for j in 0..a.cols {
        a.set(i, j, 0.0);
      }
    }

    a.clone()
  }
}