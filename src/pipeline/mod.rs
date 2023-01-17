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

impl Loader for SequentialPieline {
  fn get_name(&self) -> &str {
    "SequentialPieline"
  }

  fn get_weights(&self) -> Vec<crate::Weigths> {
    let mut weights = Vec::new();
    for layer in self.layers.iter() {
      if let Ok(l) = layer.lock() {
        if l.as_any().is::<&dyn Loader>() {
          println!("get weigths for {}", "asd");
          //weights.append(&mut l.get_weights());
        }
        if let Some(loader) =  l.as_any().downcast_ref::<&dyn Loader>() {
          println!("get weigths for {}", loader.get_name());
          //weights.append(&mut loader.get_weights());
        }
      }
    }
    weights
  }

  fn set_weights(&mut self, weights: Vec<crate::Weigths>, bias: Vec<crate::Weigths>) {
    for layer in self.layers.iter_mut() {
      if let Ok(ref mut l) = layer.lock() {
        if let Some(loader) =  l.as_mut_any().downcast_mut::<&mut dyn Loader>() {
          loader.set_weights(weights.clone(), bias.clone());
        }
      }
    }
  }
  
  fn as_any(&self) -> &dyn std::any::Any {
    self
  }
  fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
    self  
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
  
  fn as_any(&self) -> &dyn std::any::Any {
    self
  }
  fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
    self  
  }
  
}