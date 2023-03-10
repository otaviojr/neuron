pub mod cpu;

#[cfg(feature = "opencl")]
pub mod opencl;

use std::{fmt::{Display,Formatter}, time::Instant, any::Any, cell::Ref, sync::{Arc, Mutex}};
use rand::{Rng, SeedableRng};
use rand_distr::{Uniform};
use rand_xoshiro::Xoshiro256Plus;
use rand_xoshiro::rand_core::RngCore;
use crate::Neuron;

#[cfg(feature = "opencl")]
use opencl::{TensorOCL,MatrixMathOCL};

use self::{cpu::MatrixMathCPU, opencl::OCL};

pub trait MatrixMathExecutor: Any + Send + Sync {
  fn add(&self, a: &mut Tensor, b: &Tensor) -> Tensor;
  fn sub(&self, a: &mut Tensor, b: &Tensor) -> Tensor;
  fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor;
  fn mul_wise(&self, a: &mut Tensor, b: &Tensor) -> Tensor;
  fn div(&self, a: &mut Tensor, b: &Tensor) -> Tensor;
  fn dot(&self, a: &Tensor, b: &Tensor) -> f32;
  fn transpose(&self, a: &Tensor) -> Tensor;

  fn add_value(&self, a: &mut Tensor, b: f32) -> Tensor;
  fn div_value(&self, a: &mut Tensor, b: f32) -> Tensor;
  fn mul_value(&self, a: &mut Tensor, b: f32) -> Tensor;

  fn sum_row(&self, a:&Tensor) -> Tensor;
  fn broadcast(&self, a: &Tensor, rows: usize, cols: usize) -> Tensor;

  fn pad(&self, a: &Tensor, pad_row: usize, pad_col: usize) -> Tensor;

  fn zero(&self, a: &mut Tensor) -> Tensor;
}

pub enum MatrixMathExecutorEnum {
  #[cfg(feature = "opencl")]
  NONE,
  OCL(MatrixMathOCL),
  CPU(MatrixMathCPU),
}

unsafe impl Send for MatrixMathExecutorEnum {}
unsafe impl Sync for MatrixMathExecutorEnum {}

impl MatrixMathExecutorEnum {
  pub fn get_executor(&self) -> Option<Box<&dyn MatrixMathExecutor>> {
    match self {
      MatrixMathExecutorEnum::OCL(matrix_ocl) => Some(Box::new(matrix_ocl)),
      MatrixMathExecutorEnum::CPU(matrix_cpu) => Some(Box::new(matrix_cpu)),
      MatrixMathExecutorEnum::NONE => None
    }
  }
}

#[derive(Clone, Debug)]
pub struct Tensor {
  rows: usize,
  cols: usize,
  data: Vec<f32>,

  #[cfg(feature = "opencl")]
  tensor_ocl: Option<Arc<Mutex<TensorOCL>>>,
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

  pub fn new(rows: usize, cols: usize) -> Self {

    let start = Instant::now();

    let data = vec![0.0; rows * cols];

    let mut tensor_ocl = None;

    if cfg!(feature = "opencl") {
      if let Some(t) = TensorOCL::new(data.len()) {
        tensor_ocl = Some(Arc::new(Mutex::new(t)));
      }
    }

    Neuron::logger().profiling(|| format!("Zero tensor loaded after: {}ns", start.elapsed().as_nanos()));

    Tensor {
        rows,
        cols,
        data: vec![0.0; rows * cols],
        #[cfg(feature = "opencl")]
        tensor_ocl,
    }
  }

  // Initialize this tensor with random values
  pub fn random(rows: usize, cols: usize, low: f32, high: f32) -> Self {

    let start = Instant::now();

    let mut data = vec![0.0; rows * cols];
    let mut rng = rand::thread_rng();
    
    data.iter_mut().for_each(|x| *x = rng.sample(Uniform::new(low, high)));

    let mut tensor_ocl = None;

    if cfg!(feature = "opencl") {
      if let Some(t) = TensorOCL::init(&data) {
        tensor_ocl = Some(Arc::new(Mutex::new(t)));
      }
    }
 
    Neuron::logger().debug(|| format!("Random tensor loaded after: {} seconds", start.elapsed().as_secs()));

    Tensor {
      rows,
      cols,
      data,
      #[cfg(feature = "opencl")]
      tensor_ocl,
    }
  }

  pub fn randomHE(rows: usize, cols: usize, fan_in: usize) -> Self {

    let start = Instant::now();

    let mut data = vec![0.0; rows * cols];

    let he_scale = (2.0 / fan_in as f32).sqrt();
    let mut rng = Xoshiro256Plus::from_entropy();

    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let pi: f32 = std::f32::consts::PI;
    
    data.iter_mut().for_each(|x| { 
      f1 = rng.next_u64() as f32 / u64::MAX as f32;
      f2 = rng.next_u64() as f32 / u64::MAX as f32;
      *x =  ((-2.0 * f1.ln()).sqrt() * (2.0 * pi * f2).cos()) * he_scale;
    });

    let mut tensor_ocl = None;

    if cfg!(feature = "opencl") {
      if let Some(t) = TensorOCL::init(&data) {
        tensor_ocl = Some(Arc::new(Mutex::new(t)));
      }
   }

    Neuron::logger().debug(|| format!("Random tensor loaded after: {} seconds", start.elapsed().as_secs()));

    Tensor {
      rows,
      cols,
      data,
      #[cfg(feature = "opencl")]
      tensor_ocl,
    }
  }


  pub fn from_data(rows: usize, cols: usize, data: Vec<f32>) -> Self{
    assert!(data.len() == rows*cols);

    let mut tensor_ocl = None;

    if cfg!(feature = "opencl") {
      if let Some(t) = TensorOCL::init(&data) {
        tensor_ocl = Some(Arc::new(Mutex::new(t)));
      }
    }

    Tensor {
      rows,
      cols,
      data,
      #[cfg(feature = "opencl")]
      tensor_ocl,
    }
  }

  pub fn rows(&self) -> usize {
    self.rows
  }

  pub fn cols(&self) -> usize {
    self.cols
  }

  //Get a value from the tensor
  #[inline(always)]
  pub fn get(&self, row: usize, col: usize) -> f32 {
    self.data[row * self.cols + col]
  }

  //Set a tensor value
  #[inline(always)]
  pub fn set(&mut self, row: usize, col: usize, value: f32) {
    self.data[row * self.cols + col] = value;
  }

  pub fn data(&self) -> &Vec<f32> {
    self.data.as_ref()
  }

  pub fn mut_data(&mut self) -> &mut Vec<f32> {
    self.data.as_mut()
  }

  pub fn reshape(&mut self, rows: usize, cols: usize) -> Tensor {
    self.rows = rows;
    self.cols = cols;
    self.clone()
  }

  // Transpose the tensor
  pub fn transpose(&self) -> Result<Tensor, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.transpose(self));
    }
    Err(format!("No executor found"))
  }

  // Add a number to all rows and columns in the tensor
  pub fn add_value(&mut self, value: f32) -> Result<Tensor, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.add_value(self, value));
    }
    Err(format!("No executor found"))
  }

  pub fn div_value(&mut self, value: f32) -> Result<Tensor, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.div_value(self, value));
    }
    Err(format!("No executor found"))
  }

  pub fn mul_value(&mut self, value: f32) -> Result<Tensor, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.mul_value(self, value));
    }
    Err(format!("No executor found"))
  }

  //Dot Product between two tensors
  pub fn dot(&self, other: &Tensor) -> Result<f32, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.dot(&self, other));
    }
    Err(format!("No executor found"))
  }

  pub fn add(&mut self, other: &Tensor) -> Result<Tensor, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.add(self, other));
    }
    Err(format!("No executor found"))
  }

  pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.mul(&self, other));
    }
    Err(format!("No executor found"))
  }

  pub fn div(&mut self, other: &Tensor) -> Result<Tensor, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.div(self, other));
    }
    Err(format!("No executor found"))
  }

  pub fn mul_wise(&mut self, other: &Tensor) -> Result<Tensor, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.mul_wise(self, other));
    }
    Err(format!("No executor found"))
  }

  pub fn sub(&mut self, other: &Tensor) -> Result<Tensor, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.sub(self, other));
    }
    Err(format!("No executor found"))
  }

  pub fn sum_row(&self) -> Result<Tensor, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.sum_row(&self));
    }
    Err(format!("No executor found"))
  }

  pub fn broadcast(&self, rows: usize, cols: usize) -> Result<Tensor, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.broadcast(&self, rows, cols));
    }
    Err(format!("No executor found"))
  }

  pub fn pad(&self, pad_row: usize, pad_col: usize) -> Result<Tensor, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.pad(&self, pad_row, pad_col));
    }
    Err(format!("No executor found"))
  }

  pub fn zero(&mut self) -> Result<Tensor, String> {
    if let Some(executor) = Neuron::matrix().get_executor() {
      return Ok(executor.zero(self));
    }
    Err(format!("No executor found"))
  }

  pub fn sync_gpu(&mut self) {
    if cfg!(feature = "opencl") {
      self.sync_cpu_ocl();
    }
  }

  pub fn sync_cpu(&mut self) {
    if cfg!(feature = "opencl") {
      self.sync_ocl_cpu();
    }
  }
}