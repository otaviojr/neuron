pub mod math;
pub mod layers;
pub mod activations;
pub mod cost;

use std::sync::Mutex;

use math::{Tensor, MatrixMath, MatrixMathCPU};
use layers::LayerPropagation;

pub struct Neuron {
  layers: Vec<Box<Mutex<dyn LayerPropagation>>>
}

impl Neuron {
  pub fn new() -> Self {
    Neuron {
      layers: Vec::new()
    }
  }

  pub fn forward(&self, input: Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {
    let mut i = Some(input);
    for layer in self.layers.iter() {
      if let Some(ref i1) = i {
        i = layer.lock().unwrap().forward(i1);
      }
    }
    i
  }

  pub fn backward(&self, input: Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {
    let mut i = Some(input);
    for (index,layer) in self.layers.iter().rev().enumerate() {
      if let Some(ref i1) = i {
        i = layer.lock().unwrap().backward(i1, index==0);
      }
    }
    i
  }

  pub fn matrix_math() -> Box<dyn MatrixMath> {
    Box::new(MatrixMathCPU { })
  }

  pub fn add_layer(&mut self, layer: Box<Mutex<dyn LayerPropagation>>) -> &mut Self {
    self.layers.push(layer);
    self
  }
}
