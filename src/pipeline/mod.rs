use std::sync::{Mutex, Arc};
use std::any::Any;

use crate::{Propagation, math::Tensor, Loader};

pub struct SequentialPieline{
  layers: Vec<Mutex<Box<dyn Any>>>
} 

impl SequentialPieline {
  pub fn new() -> Self {
    SequentialPieline {
      layers: Vec::new()
    }
  }

  pub fn add_layer(&mut self, layer: Mutex<Box<dyn Any>>) -> &mut Self {
    self.layers.push(layer);
    self
  }
}

impl Loader for SequentialPieline {
  fn get_weights(&self) -> Vec<crate::Weigths> {
    let mut weights = Vec::new();
    for layer in self.layers.iter() {
      if let Ok(l) = layer.lock() {
        if let Some(loader) =  l.downcast_ref::<Box<dyn Loader>>() {
          //weights.append(&mut loader.get_weights());
        }
      }
    }
    weights
  }

  fn set_weights(&mut self, weights: Vec<crate::Weigths>, bias: Vec<crate::Weigths>) {
    for layer in self.layers.iter_mut() {
      if let Ok(ref mut l) = layer.lock() {
        if let Some(loader) =  l.downcast_mut::<Box<dyn Loader>>() {
          loader.set_weights(weights.clone(), bias.clone());
        }
      }
    }
  }
}

impl Propagation for SequentialPieline {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {
    let mut i = Some(input.clone());
    for layer in self.layers.iter() {
      if let Some(ref i1) = i {
        if let Ok(ref mut l) = layer.lock() {
          if let Some(prop) =  l.downcast_mut::<Box<dyn Propagation>>() {
            i = prop.forward(i1);
          }
        }
      }
    }
    i
  }

  fn backward(&mut self, input: &Vec<Box<Tensor>>, first: bool) -> Option<Vec<Box<Tensor>>> {
    let mut i = Some(input.clone());
    for (index,layer) in self.layers.iter().rev().enumerate() {
      if let Some(ref i1) = i {
        if let Ok(ref mut l) = layer.lock() {
          if let Some(prop) =  l.downcast_mut::<Box<dyn Propagation>>() {
            i = prop.backward(i1, index == 0 && first);
          }
        }
      }
    }
    i
  }
}