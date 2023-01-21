pub mod cpu;
pub mod opencl;

use std::{fmt::{Display,Formatter}, time::Instant};
use std::ops::{Div};
use rand::{Rng, SeedableRng};
use rand_distr::{Uniform};
use rand_xoshiro::Xoshiro256Plus;
use rand_xoshiro::rand_core::RngCore;
use crate::Neuron;

pub trait MatrixMathExecutor {
  fn add(&self, a: &Tensor, b: &Tensor) -> Tensor;
  fn sub(&self, a: &Tensor, b: &Tensor) -> Tensor;
  fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor;
  fn mul_wise(&self, a: &Tensor, b: &Tensor) -> Tensor;
  fn div(&self, a: &Tensor, b: &Tensor) -> Tensor;
  fn dot(&self, a: &Tensor, b: &Tensor) -> f64;
  fn transpose(&self, a: &Tensor) -> Tensor;

  fn add_value(&self, a: &Tensor, b: f64) -> Tensor;
  fn div_value(&self, a: &Tensor, b: f64) -> Tensor;
  fn mul_value(&self, a: &Tensor, b: f64) -> Tensor;

  fn sum_row(&self, a:&Tensor) -> Tensor;
  fn broadcast(&self, a: &Tensor, rows: usize, cols: usize) -> Tensor;

  fn pad(&self, a: &Tensor, pad_row: usize, pad_col: usize) -> Tensor;
}

#[derive(Clone, Debug)]
pub struct Tensor {
  rows: usize,
  cols: usize,
  
  data: Vec<f64>,
}

impl Display for Tensor {
  fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
    writeln!(f, "[")?;
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
  pub fn random(rows: usize, cols: usize, low: f64, high: f64) -> Self {
    let mut data = vec![0.0; rows * cols];
    let mut rng = rand::thread_rng();
    
    let start = Instant::now();
    data.iter_mut().for_each(|x| *x = rng.sample(Uniform::new(low, high)));
    let elapsed = start.elapsed();
    println!("Random tensor loaded after: {} seconds", elapsed.as_secs());

    Tensor {
      rows,
      cols,
      data,
    }
  }

  pub fn randomHE(rows: usize, cols: usize, fan_in: usize) -> Self {
    let mut data = vec![0.0; rows * cols];

    let he_scale = (2.0 / fan_in as f64).sqrt();
    let mut rng = Xoshiro256Plus::from_entropy();

    let start = Instant::now();
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let pi: f64 = std::f64::consts::PI;
    data.iter_mut().for_each(|x| { 
      f1 = rng.next_u64() as f64 / u64::MAX as f64;
      f2 = rng.next_u64() as f64 / u64::MAX as f64;
      *x =  ((-2.0 * f1.ln()).sqrt() * (2.0 * pi * f2).cos()) * he_scale;
    });
    let elapsed = start.elapsed();
    println!("Random tensor loaded after: {} seconds", elapsed.as_secs());

    Tensor {
      rows,
      cols,
      data,
    }
  }


  pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> Self{
    assert!(data.len() == rows*cols);
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
    self.cols
  }

  //Get a value from the tensor
  pub fn get(&self, row: usize, col: usize) -> f64 {
    self.data[row * self.cols + col]
  }

  //Set a tensor value
  pub fn set(&mut self, row: usize, col: usize, value: f64) {
    self.data[row * self.cols + col] = value;
  }

  pub fn data(&self) -> &Vec<f64> {
    self.data.as_ref()
  }

  // Transpose the tensor
  pub fn transpose(&self) -> Tensor {
    return Neuron::matrix().lock().unwrap().as_ref().unwrap().transpose(self);
  }

  // Add a number to all rows and columns in the tensor
  pub fn add_value(&self, value: f64) -> Tensor {
    return Neuron::matrix().lock().unwrap().as_ref().unwrap().add_value(&self, value);
  }

  pub fn div_value(&self, value: f64) -> Tensor {
    return Neuron::matrix().lock().unwrap().as_ref().unwrap().div_value(&self, value);
  }

  pub fn mul_value(&self, value: f64) -> Tensor {
    return Neuron::matrix().lock().unwrap().as_ref().unwrap().mul_value(&self, value);
  }

  //Dot Product between two tensors
  pub fn dot(&self, other: &Tensor) -> f64 {
    return Neuron::matrix().lock().unwrap().as_ref().unwrap().dot(self, other);
  }

  pub fn add(&self, other: &Tensor) -> Tensor {
    return Neuron::matrix().lock().unwrap().as_ref().unwrap().add(&self, other);
  }

  pub fn mul(&self, other: &Tensor) -> Tensor {
    return Neuron::matrix().lock().unwrap().as_ref().unwrap().mul(&self, other);
  }

  pub fn mul_wise(&self, other: &Tensor) -> Tensor {
    return Neuron::matrix().lock().unwrap().as_ref().unwrap().mul_wise(&self, other);
  }

  pub fn sub(&self, other: &Tensor) -> Tensor {
    return Neuron::matrix().lock().unwrap().as_ref().unwrap().sub(&self, other);
  }

  pub fn sum_row(&self) -> Tensor {
    return Neuron::matrix().lock().unwrap().as_ref().unwrap().sum_row(&self);
  }

  pub fn broadcast(&self, rows: usize, cols: usize) -> Tensor {
    return Neuron::matrix().lock().unwrap().as_ref().unwrap().broadcast(&self, rows, cols);
  }

  pub fn pad(&self, pad_row: usize, pad_col: usize) -> Tensor {
    return Neuron::matrix().lock().unwrap().as_ref().unwrap().pad(&self, pad_row, pad_col);
  }
}

//Divide two tensors
impl Div for Tensor {
  type Output = Tensor;

  fn div(self, other: Tensor) -> Tensor {
    return Neuron::matrix().lock().unwrap().as_ref().unwrap().div(&self, &other);
  }
}
