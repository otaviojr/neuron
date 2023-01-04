pub mod math;
pub mod layers;
pub mod activations;
pub mod cost;

use math::{Tensor, MatrixMath, MatrixMathCPU};
use layers::Layer;

pub struct Neuron {
  layers: Vec<Box<dyn Layer>>
}

impl Neuron {
  pub fn new() -> Self {
    Neuron {
      layers: Vec::new()
    }
  }

  pub fn forward(&self, input: Tensor) -> Option<Tensor> {
    let mut i = Some(input);
    if let Some(ref i0) = i {
      println!("Input layer size = {}x{}", i0.rows(), i0.cols());
    }
    for layer in self.layers.iter() {
      if let Some(ref i1) = i {
        i = layer.forward(i1);
        if let Some(ref i2) = i {
          println!("Hidden layer size = {}x{}", i2.rows(), i2.cols());
        }
      }
    }
    i
  }

  pub fn backward(&self, input: Tensor) -> Option<Tensor> {
    let mut i = Some(input);
    if let Some(ref i0) = i {
      println!("Input layer size = {}x{}", i0.rows(), i0.cols());
    }
    for layer in self.layers.iter() {
      if let Some(ref i1) = i {
        i = layer.backward(i1);
        if let Some(ref i2) = i {
          println!("Hidden layer size = {}x{}", i2.rows(), i2.cols());
        }
      }
    }
    i
  }

  pub fn matrix_math() -> Box<dyn MatrixMath> {
    Box::new(MatrixMathCPU { })
  }

  pub fn add_layer(&mut self, layer: Box<dyn Layer>) -> &mut Self {
    self.layers.push(layer);
    self
  }
}
