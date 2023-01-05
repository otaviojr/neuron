use rand::{Rng};
use rand_distr::{Uniform};

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
      bias: rng.sample(Uniform::new(-0.0000000000005, 0.0000000000005)),
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
    println!("Bias = {}", self.bias);
    let z1 = self.weights.mul(input).add_value(self.bias);
    //println!("z1 = {}", z1);
    let ret = Some(self.activation.forward(&z1));
    self.last_z1 = Some(z1);
    self.last_input = Some(Tensor::from_data(input.rows(), input.cols(), input.data().to_owned()));
    
    ret
  }

  fn backward(&mut self, input: &Tensor) -> Option<Tensor> {
    if let Some(ref z1) = self.last_z1 {
      let dz = self.weights.transpose().mul(&input).mul_wise(&self.activation.backward(z1));
      println!("dz size = {}x{}", dz.rows(), dz.cols());
      //println!("dz = {}", dz);
      if let Some(ref forward_input) = self.last_input {
        println!("forward_input size = {}x{}", forward_input.rows(), forward_input.cols());
        let dw = dz.mul(&forward_input.transpose()).div_value(input.cols() as f64);
        println!("dw size = {}x{}", dw.rows(), dw.cols());
        //println!("dw = {}", dw);
        let db = Tensor::from_data(dz.rows(), dz.cols(), dz.data().to_owned());
        println!("db size = {}x{}", db.rows(), db.cols());

        self.weights = self.weights.add(&dw.mul_value(0.1));
        let dbf = db.data().to_owned().into_iter().reduce(|count, value| count + value ).unwrap() / input.cols() as f64;
        self.bias += dbf * 0.01;

        return Some(self.weights.transpose().mul(&dz))
      }
    }
    None
  }
}