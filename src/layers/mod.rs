use crate::math::{Tensor, MatrixMathCPU};
use crate::activations::Activation;

pub trait Layer {
  fn forward(&self, input: &Tensor) -> Tensor;
  fn backward(&self, input: &Tensor) -> Tensor;
}

pub struct LinearLayer {
  activation: Box<dyn Activation>
}

impl LinearLayer {
  pub fn new(activation: Box<dyn Activation>) -> Self {
    LinearLayer {
      activation: activation
    }
  }
}

impl Layer for LinearLayer {
  fn forward(&self, input: &Tensor) -> Tensor {
    Tensor::random(0, 0)
  }

  fn backward(&self, input: &Tensor) -> Tensor {
    Tensor::random(0, 0)
  }
}