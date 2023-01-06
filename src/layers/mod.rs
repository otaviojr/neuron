use rand::{Rng};
use rand_distr::{Uniform};

use crate::math::Tensor;
use crate::activations::Activation;

pub trait Layer {
  fn get_nodes(&self) -> usize;
  fn forward(&mut self, input: &Tensor) -> Option<Tensor>;
  fn backward(&mut self, input: &Tensor, first: bool) -> Option<Tensor>;
}

pub struct LinearLayer {
  activation: Box<dyn Activation>,
  nodes: usize,
  weights: Tensor,
  bias: Tensor,

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
      bias: Tensor::zeros(nodes,1),
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
    //println!("Input = {}", input);
    //println!("Weights = {}", self.weights);
    let z1_1 = self.weights.mul(input);
    println!("z1_1 = {}x{}", z1_1.rows(), z1_1.cols());
    let b_bias = self.bias.broadcast(z1_1.rows(), z1_1.cols());
    println!("b_bias = {}x{}", b_bias.rows(), b_bias.cols());
    let z1 = z1_1.add(&b_bias);

    //println!("z1 = {}", z1);
    let ret = Some(self.activation.forward(&z1));
    self.last_z1 = Some(z1);
    self.last_input = Some(Tensor::from_data(input.rows(), input.cols(), input.data().to_owned()));
    
    ret
  }

  fn backward(&mut self, input: &Tensor, first: bool) -> Option<Tensor> {
    if let Some(ref z1) = self.last_z1 {
      let dz;
      if first {
        dz = Tensor::from_data(input.rows(), input.cols(), input.data().to_owned());
      } else {
        dz = input.mul_wise(&self.activation.backward(z1));
      }
      println!("dz size = {}x{}", dz.rows(), dz.cols());
      //println!("dz = {}", dz);
      if let Some(ref forward_input) = self.last_input {
        println!("forward_input size = {}x{}", forward_input.rows(), forward_input.cols());
        let dw = dz.mul(&forward_input.transpose()).div_value(forward_input.cols() as f64);
        println!("dw size = {}x{}", dw.rows(), dw.cols());
        //println!("dw = {}", dw);
        let db = Tensor::from_data(dz.rows(), dz.cols(), dz.data().to_owned());
        println!("db size = {}x{}", db.rows(), db.cols());

        let ret = Some(self.weights.transpose().mul(&dz));

        self.weights = self.weights.sub(&dw.mul_value(0.1));
        let dbf = db.sum_row().div_value(forward_input.cols() as f64);
        self.bias = self.bias.sub(&dbf.mul_value(0.1));

        return ret;
      }
    }
    None
  }
}