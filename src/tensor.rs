use std::{error::Error, fmt::{Display,Formatter}};
use std::ops::Mul;

use rand_distr::{Normal, Distribution};

#[derive(Debug)]
pub struct OperationError {
    message: String,
}

impl OperationError {
    fn new(message: &str) -> OperationError {
        OperationError {
            message: message.to_string(),
        }
    }
}

impl Error for OperationError {}

impl Display for OperationError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

pub struct Tensor {
  rows: usize,
  cols: usize,
  
  data: Vec<f64>,
}

impl Display for Tensor {
  fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
      for i in 0..self.rows {
          for j in 0..self.cols {
              write!(f, "{} ", self.get(i, j))?;
          }
          writeln!(f)?;
      }

      Ok(())
  }
}

impl Tensor {

  pub fn new(rows: usize, cols: usize) -> Tensor {
    Tensor {
        rows,
        cols,
        data: vec![0.0; rows * cols],
    }
  }

  pub fn random(&mut self) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();
    
    for i in 0..self.rows {
      for j in 0..self.cols {
        self.set(i, j, normal.sample(&mut rng));
      }
    }
  }

  pub fn get(&self, row: usize, col: usize) -> f64 {
    self.data[row * self.cols + col]
  }

  pub fn set(&mut self, row: usize, col: usize, value: f64) {
    self.data[row * self.cols + col] = value;
  }

  // A method to transpose the matrix
  pub fn transpose(&self) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::new(self.cols, self.rows);

    // Transpose the matrix
    for i in 0..self.rows {
        for j in 0..self.cols {
            result.set(j, i, self.get(i, j));
        }
    }

    result
  }

  // A method to add a number to all rows and columns in the matrix
  pub fn add(&mut self, value: f64) {
    for i in 0..self.rows {
      for j in 0..self.cols {
        let val = self.get(i, j);
        self.set(i, j, val + value);
      }
    }
  }

  pub fn dot(&self, other: &Tensor) -> Result<f64, OperationError> {
    // Check that the matrices are compatible for the dot product
    if self.rows != other.rows || self.cols != other.cols {
        return Err(OperationError::new("matrices are not compatible for dot product"));
    }

    let mut result = 0.0;

    // Calculate the dot product
    for i in 0..self.rows {
        for j in 0..self.cols {
            result += self.get(i, j) * other.get(i, j);
        }
    }

    Ok(result)
  }
}

impl Mul for Tensor {
  type Output = Tensor;

  fn mul(self, other: Tensor) -> Tensor {
    // Check that the matrices are compatible for multiplication
    if self.cols != other.rows {
      return Tensor::new(0,0);
    }

    // Create a new tensor to store the result
    let mut result = Tensor::new(self.rows, other.cols);

    // Perform matrix multiplication
    for i in 0..self.rows {
      for j in 0..other.cols {
        let mut sum = 0.0;
        for k in 0..self.cols {
          sum += self.get(i, k) * other.get(k, j);
        }
        result.set(i, j, sum);
      }
    }

    result
  }
}