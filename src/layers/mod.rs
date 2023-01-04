use std::ops::Add;

use rand::{Rng};
use rand_distr::StandardNormal;

use crate::math::Tensor;
use crate::activations::Activation;

pub trait Layer {
  fn get_nodes(&self) -> usize;
  fn forward(&self, input: &Tensor) -> Option<Tensor>;
  fn backward(&self, input: &Tensor) -> Option<Tensor>;
}

pub struct LinearLayer {
  activation: Box<dyn Activation>,
  nodes: usize,
  weights: Tensor,
  bias: f64
}

impl LinearLayer {
  pub fn new(input_size: usize, nodes: usize, activation: Box<dyn Activation>) -> Self {
    let mut rng = rand::thread_rng();

    LinearLayer {
      activation: activation,
      nodes: nodes,
      weights: Tensor::random(nodes,input_size),
      bias: rng.sample(StandardNormal)
    }
  }
}

impl Layer for LinearLayer {
  fn get_nodes(&self) -> usize {
    return self.nodes;
  }

  fn forward(&self, input: &Tensor) -> Option<Tensor> {
    println!("Layer weights size = {}x{}", self.weights.rows(), self.weights.cols());
    let z1 = self.weights.mul(input).add_value(self.bias);
    Some(self.activation.forward(&z1))
  }

  fn backward(&self, input: &Tensor) -> Option<Tensor> {
    None
  }
}