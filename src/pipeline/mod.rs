use std::sync::{Mutex, Arc};
use std::any::Any;

use crate::{Propagation, math::Tensor, Loader};

pub struct SequentialPieline{
  layers: Vec<Mutex<Box<dyn Propagation>>>
} 

impl SequentialPieline {
  pub fn new() -> Self {
    SequentialPieline {
      layers: Vec::new()
    }
  }

  pub fn add_layer(&mut self, layer: Mutex<Box<dyn Propagation>>) -> &mut Self {
    self.layers.push(layer);
    self
  }
}

unsafe impl Send for SequentialPieline {}
unsafe impl Sync for SequentialPieline {}

impl Loader for SequentialPieline {
  fn get_name(&self) -> String {
    "SequentialPieline".to_owned()
  }

  fn get_weights(&self) -> Vec<crate::Weigths> {
    let mut weights = Vec::new();
    for layer in self.layers.iter() {
      if let Ok(l) = layer.lock() {
        if let Some(loader) =  l.as_loader() {
          weights.extend(loader.get_weights());
        }
      }
    }
    weights
  }

  fn set_weights(&mut self, weights: Vec<crate::Weigths>) {
    for layer in self.layers.iter_mut() {
      if let Ok(ref mut l) = layer.lock() {
        if let Some(loader) =  l.as_mut_loader() {
          loader.set_weights(weights.clone());
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
          i = l.forward(i1);
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
          i = l.backward(i1, index == 0 && first);
        }
      }
    }
    i
  }
  
  fn as_loader(&self) -> Option<&dyn Loader> {
    Some(self)
  }

  fn as_mut_loader(&mut self) -> Option<&mut dyn Loader> {
    Some(self)
  }

  
}