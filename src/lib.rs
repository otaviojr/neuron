pub mod math;
pub mod layers;
pub mod activations;

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
    for layer in self.layers.iter() {
      if let Some(i1) = i {
        i = layer.forward(&i1);
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
