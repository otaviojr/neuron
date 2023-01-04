use std::{fmt::{Display,Formatter}, time::Instant};
use std::ops::{Mul, Add, Sub, Div};
use rand::{Rng};
use rand_distr::StandardNormal;
use crate::Neuron;

pub trait MatrixMath {
  fn add(&self, a: &Tensor, b: &Tensor) -> Tensor;
  fn sub(&self, a: &Tensor, b: &Tensor) -> Tensor;
  fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor;
  fn div(&self, a: &Tensor, b: &Tensor) -> Tensor;
  fn dot(&self, a: &Tensor, b: &Tensor) -> f64;
  fn transpose(&self, a: &Tensor) -> Tensor;

  fn add_value(&self, a: &Tensor, b: f64) -> Tensor;
}

#[derive(Copy, Clone)]
pub struct MatrixMathCPU;

impl MatrixMath for MatrixMathCPU {
  fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
      // Check that the tensors are the same size
      if a.cols != b.cols || a.rows != b.rows {
        return Tensor::zeros(0,0);
      }

      // Create a new tensor to store the result
      let mut result = Tensor::zeros(a.rows, a.cols);

      // Perform element-wise addition
      for i in 0..a.data.len() {
          result.data[i] = a.data[i] + b.data[i];
      }

      result
  }

  fn sub(&self, a: &Tensor, b: &Tensor) -> Tensor {
      // Check that the tensors are the same size
      if a.cols != b.cols || a.rows != b.rows {
        return Tensor::zeros(0,0);
      }

      // Create a new tensor to store the result
      let mut result = Tensor::zeros(a.rows, a.cols);

      // Perform element-wise subtraction
      for i in 0..a.data.len() {
          result.data[i] = a.data[i] - b.data[i];
      }

      result
  }

  fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
    // Check that the tensors are compatible for multiplication
    if a.cols != b.rows {
      return Tensor::zeros(0,0);
    }

    // Create a new tensor to store the result
    let mut result = Tensor::zeros(a.rows, b.cols);

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

  fn div(&self, a: &Tensor, b: &Tensor) -> Tensor {
      // Check that the tensors are the same size
      if a.cols != b.cols || a.rows != b.rows {
        return Tensor::zeros(0,0);
      }

      // Create a new tensor to store the result
      let mut result = Tensor::zeros(a.rows, a.cols);

      // Perform element-wise division
      for i in 0..a.data.len() {
          result.data[i] = a.data[i] / b.data[i];
      }

      result
  }

  fn dot(&self, a: &Tensor, b: &Tensor) -> f64 {
    let mut result = 0.0;

    // Check that the tensors are compatible for the dot product
    if a.rows != b.rows || a.cols != b.cols {
        return result;
    }

    // Calculate the dot product
    for i in 0..a.rows {
        for j in 0..a.cols {
            result += a.get(i, j) * b.get(i, j);
        }
    }

    result
  }

  fn transpose(&self, a: &Tensor) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::zeros(a.cols, a.rows);

    // Transpose the matrix
    for i in 0..a.rows {
        for j in 0..a.cols {
            result.set(j, i, a.get(i, j));
        }
    }

    result
  }

  fn add_value(&self, a: &Tensor, value: f64) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::zeros(a.rows, a.cols);

    for i in 0..a.rows {
      for j in 0..a.cols {
        let val = a.get(i, j);
        result.set(i, j, val + value);
      }
    }

    result
  }

}

pub struct Tensor {
  rows: usize,
  cols: usize,
  
  data: Vec<f64>,
}

impl Display for Tensor {
  fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
    write!(f, "[")?;
    for i in 0..self.rows {
      write!(f, "[")?;
      for j in 0..self.cols {
              write!(f, "{}", self.get(i, j))?;
              if j < self.cols-1 {
                write!(f," ")?;
              }
          }
          if i < self.rows-1 {
            writeln!(f,"]")?;
          } else {
            write!(f,"]")?;
          }
      }
      writeln!(f, "]")?;

      Ok(())
  }
}

impl Tensor {

  pub fn zeros(rows: usize, cols: usize) -> Self {
    Tensor {
        rows,
        cols,
        data: vec![0.0; rows * cols],
    }
  }

  // Initialize this tensor with random values
  pub fn random(rows: usize, cols: usize) -> Self {
    let mut data = vec![0.0; rows * cols];
    let mut rng = rand::thread_rng();
    
    let start = Instant::now();
    data.iter_mut().for_each(|x| *x = rng.sample(StandardNormal));
    let elapsed = start.elapsed();
    println!("Matrix loaded after: {} seconds", elapsed.as_secs());

    Tensor {
      rows,
      cols,
      data,
    }
  }

  pub fn rows(&self) -> usize {
    self.rows
  }

  pub fn cols(&self) -> usize {
    self.rows
  }

  //Get a value from the tensor
  pub fn get(&self, row: usize, col: usize) -> f64 {
    self.data[row * self.cols + col]
  }

  //Set a tensor value
  pub fn set(&mut self, row: usize, col: usize, value: f64) {
    self.data[row * self.cols + col] = value;
  }

  // Transpose the tensor
  pub fn transpose(&self) -> Tensor {
    return Neuron::matrix_math().transpose(self);
  }

  // Add a number to all rows and columns in the tensor
  pub fn add_value(&mut self, value: f64) -> Tensor {
    return Neuron::matrix_math().add_value(&self, value);
  }

  //Dot Product between two tensors
  pub fn dot(&self, other: &Tensor) -> f64 {
    return Neuron::matrix_math().dot(self, other);
  }
}

//Add two tensors
impl Add for Tensor {
  type Output = Tensor;

  fn add(self, other: Tensor) -> Tensor {
    return Neuron::matrix_math().add(&self, &other);
  }
}

//Subtract two tensors
impl Sub for Tensor {
  type Output = Tensor;

  fn sub(self, other: Tensor) -> Tensor {
    return Neuron::matrix_math().sub(&self, &other);
  }
}

//Multiply two tensors
impl Mul for Tensor {
  type Output = Tensor;

  fn mul(self, other: Tensor) -> Tensor {
    return Neuron::matrix_math().mul(&self, &other);
  }
}

//Divide two tensors
impl Div for Tensor {
  type Output = Tensor;

  fn div(self, other: Tensor) -> Tensor {
    return Neuron::matrix_math().div(&self, &other);
  }
}