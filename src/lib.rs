pub mod math;
pub mod layers;
pub mod activations;
pub mod cost;
pub mod pipeline;

use std::{sync::Mutex, any::Any};

use math::{Tensor, MatrixMath, MatrixMathCPU};

pub trait Propagation: Any {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>>;
  fn backward(&mut self, input: &Vec<Box<Tensor>>, first: bool) -> Option<Vec<Box<Tensor>>>;
  fn as_loader(&self) -> Option<&dyn Loader>;
  fn as_mut_loader(&mut self) -> Option<&mut dyn Loader>;
}

#[derive(Clone)]
pub struct Weigths {
  name: &'static str,
  weights: Vec<Box<Tensor>>,
  bias: Vec<Box<Tensor>>
}

pub trait Loader: Any {
  fn get_name(&self) -> &str;
  fn get_weights(&self) -> Vec<Weigths>;
  fn set_weights(&mut self, weights: Vec<Weigths>, bias: Vec<Weigths>);
}

pub struct Neuron {
  pipelines: Vec<Mutex<Box<dyn Propagation>>>
}

impl Neuron {
  pub fn new() -> Self {
    Neuron {
      pipelines: Vec::new()
    }
  }

  pub fn forward(&self, input: Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {
    let mut i = Some(input);
    for layer in self.pipelines.iter() {
      if let Some(ref i1) = i {
        i = layer.lock().unwrap().forward(i1);
      }
    }
    i
  }

  pub fn backward(&self, input: Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {
    let mut i = Some(input);
    for (index,layer) in self.pipelines.iter().rev().enumerate() {
      if let Some(ref i1) = i {
        i = layer.lock().unwrap().backward(i1, index==0);
      }
    }
    i
  }

  pub fn matrix_math() -> Box<dyn MatrixMath> {
    Box::new(MatrixMathCPU { })
  }

  pub fn add_pipeline(&mut self, layer: Mutex<Box<dyn Propagation>>) -> &mut Self {
    self.pipelines.push(layer);
    self
  }
}
