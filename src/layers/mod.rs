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
  bias: Tensor
}

impl LinearLayer {
  pub fn new(input_size: usize, nodes: usize, activation: Box<dyn Activation>) -> Self {
    LinearLayer {
      activation: activation,
      nodes: nodes,
      weights: Tensor::random(nodes,input_size),
      bias: Tensor::random(nodes,1)
    }
  }
}

impl Layer for LinearLayer {
  fn get_nodes(&self) -> usize {
    return self.nodes;
  }

  fn forward(&self, input: &Tensor) -> Option<Tensor> {
    None
  }

  fn backward(&self, input: &Tensor) -> Option<Tensor> {
    None
  }
}