pub mod math;
pub mod layers;
pub mod activations;

use math::{MatrixMath, MatrixMathCPU};
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

  pub fn matrix_math() -> Box<dyn MatrixMath> {
    Box::new(MatrixMathCPU { })
  }

  pub fn add_layer(&mut self, layer: Box<dyn Layer>) -> &mut Self {
    self.layers.push(layer);
    self
  }
}
