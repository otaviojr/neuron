use std::ops::Add;

use rand::{Rng};
use rand_distr::StandardNormal;

use crate::math::Tensor;
use crate::activations::Activation;

pub trait Layer {
  fn get_nodes(&self) -> usize;
  fn forward(&mut self, input: &Tensor) -> Option<Tensor>;
  fn backward(&mut self, input: &Tensor) -> Option<Tensor>;
}

pub struct LinearLayer {
  activation: Box<dyn Activation>,
  nodes: usize,
  weights: Tensor,
  bias: f64,

  last_input: Option<Tensor>,
  last_z1: Option<Tensor>
}

impl LinearLayer {
  pub fn new(input_size: usize, nodes: usize, activation: Box<dyn Activation>) -> Self {
    let mut rng = rand::thread_rng();

    LinearLayer {
      activation: activation,
      nodes: nodes,
      weights: Tensor::random(nodes,input_size),
      bias: rng.sample(StandardNormal),
      last_input: None,
      last_z1: None
    }
  }
}

impl Layer for LinearLayer {
  fn get_nodes(&self) -> usize {
    return self.nodes;
  }

  fn forward(&mut self, input: &Tensor) -> Option<Tensor> {
    println!("Layer weights size = {}x{}", self.weights.rows(), self.weights.cols());
    let z1 = self.weights.mul(input).add_value(self.bias);
    let ret = Some(self.activation.forward(&z1));

    self.last_z1 = Some(z1);
    self.last_input = Some(Tensor::from_data(input.rows(), input.cols(), input.data().to_owned()));
    
    ret
  }

  fn backward(&mut self, input: &Tensor) -> Option<Tensor> {
    if let Some(ref z1) = self.last_z1 {
      let dz = input.mul_wise(&self.activation.backward(z1));
      println!("dz size = {}x{}", dz.rows(), dz.cols());
      if let Some(ref input) = self.last_input {
        let dw = dz.mul(&input.transpose());
        println!("dw size = {}x{}", dw.rows(), dw.cols());
        let db = Tensor::from_data(dz.rows(), dz.cols(), dz.data().to_owned());
        println!("db size = {}x{}", db.rows(), db.cols());
        return Some(self.weights.transpose().mul(&dz))
      }
    }
    None
  }
}