use math::{Tensor,MatrixMathCPU};

pub mod math;

pub struct Neuron {

}

impl Neuron {
  pub fn new() -> Self {
    Neuron {

    }
  }

  pub fn new_random_tensor(&self, rows: usize, cols: usize) -> Tensor {
    Tensor::random(rows, cols, Box::new(MatrixMathCPU::new()))
  }

  pub fn new_tensor(&self, rows: usize, cols: usize) -> Tensor {
    Tensor::zeros(rows, cols, Box::new(MatrixMathCPU::new()))
  }
}