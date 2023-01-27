pub mod cpu;

#[cfg(feature = "opencl")]
pub mod opencl;

use std::sync::Arc;
use std::time::Instant;

use crate::{Propagation, Loader, Weigths, Neuron};
use crate::math::Tensor;
use crate::activations::Activation;

pub struct DenseLayerConfig {
  pub activation: Arc<dyn Activation>,
  pub learn_rate: f32,
}

pub struct DenseLayer {
  name: String,
  config: DenseLayerConfig,
  weights: Tensor,
  bias: Tensor,

  last_input: Option<Vec<Box<Tensor>>>,
  last_z1: Option<Tensor>
}

pub trait DenseLayerExecutor {
  fn forward(&self, input: &Vec<Box<Tensor>>, weights: &Tensor, bias: &Tensor, config: &DenseLayerConfig) -> Option<(Vec<Box<Tensor>>, Tensor, Vec<Box<Tensor>>)>;
  fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, last_z1: &Tensor, weights: &mut Tensor, bias: &mut Tensor, activate: bool, config: &DenseLayerConfig) -> Option<Vec<Box<Tensor>>>;
}

impl DenseLayer {
  pub fn new(name: String, input_size: usize, nodes: usize, config: DenseLayerConfig) -> Self {
    DenseLayer {
      name,
      weights: Tensor::randomHE(nodes,input_size, input_size),
      bias: Tensor::randomHE(nodes,1, input_size),
      last_input: None,
      last_z1: None,
      config: config,
    }
  }
}

impl Loader for DenseLayer {
  fn get_name(&self) -> String {
    self.name.clone()
  }

  fn get_weights(&self) -> Vec<Weigths> {
    vec![Weigths {
      name: self.name.clone(),
      weights: vec![Box::new(self.weights.clone())],
      bias: vec![Box::new(self.bias.clone())]
    }]
  }

  fn set_weights(&mut self, weights: Vec<Weigths>) {
    for w in weights {
      if w.name == self.name {
        self.weights = *w.weights[0].clone();
        self.bias = *w.bias[0].clone();
      }
    }
  }
}

impl Propagation for DenseLayer {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {

    if let Some((forward_input,z1,ret)) = Neuron::executors().dense.forward(input, &self.weights, &self.bias, &self.config) {
      self.last_z1 = Some(z1.clone());
      self.last_input = Some(forward_input.clone());
      return Some(ret);
    }
    None
  }

  fn backward(&mut self, input: &Vec<Box<Tensor>>, first: bool) -> Option<Vec<Box<Tensor>>> {
    if let Some(ref forward_input) = self.last_input {
      if let Some(ref z1) = self.last_z1{
        if let Some(ret) = Neuron::executors().dense.backward(input, forward_input, z1, &mut self.weights, &mut self.bias, !first, &self.config) {
          self.last_z1 = Some(z1.clone());
          self.last_input = Some(forward_input.clone());
          return Some(ret);
        }
      }
    }
    None
  }
  
  
  fn as_loader(&self) -> Option<&dyn Loader> {
    Some(self)
  }

  fn as_mut_loader(&mut self) -> Option<&mut dyn Loader> {
    Some(self)  
  }
}

#[derive(Clone)]
pub struct ConvLayerConfig {
  pub activation: Arc<dyn Activation + Send + Sync>,
  pub learn_rate: f32,
  pub padding: usize,
  pub stride: usize,
}

pub trait ConvLayerExecutor {
  fn forward(&self, input: &Vec<Box<Tensor>>, filters: Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: Vec<f32>, config: &ConvLayerConfig) -> Option<(Vec<Box<Tensor>>, Vec<Box<Tensor>>, Vec<Box<Tensor>>)>;
  fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, last_z1: &Vec<Box<Tensor>>, filters: &mut Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: &mut Vec<f32>, activate: bool, config: &ConvLayerConfig) -> Option<Vec<Box<Tensor>>>;
}

pub struct ConvLayer {
  name: String,
  config: ConvLayerConfig,
  filters: Vec<Vec<Tensor>>,
  filter_size: (usize,usize),
  bias: Vec<f32>,

  last_input: Option<Vec<Box<Tensor>>>,
  last_z1: Option<Vec<Box<Tensor>>>
}

impl ConvLayer {
  pub fn new(name: String, n_channels: usize, n_filters: usize, filter_size: (usize,usize), config: ConvLayerConfig) -> Self {
    let mut filters = Vec::new();
    for i in 0 .. n_filters {
      let mut filter_channels = Vec::new();
      for j in 0 .. n_channels {
        let filter_channel = Tensor::randomHE(filter_size.0, filter_size.1, n_filters);
        filter_channels.push(filter_channel);
      }
      filters.push(filter_channels);
    }
  
    ConvLayer {
      name,
      filters,
      filter_size,
      bias: vec![0.0; n_filters],
      config,

      last_input: None,
      last_z1: None
    }
  }
}

//implement the loader trait
impl Loader for ConvLayer {
  fn get_name(&self) -> String {
    self.name.clone()
  }

  fn get_weights(&self) -> Vec<Weigths> {    
    let mut final_weigths = Vec::new();

    for (idx, filter )in self.filters.iter().enumerate() {
      let mut weigths = Vec::new();
      for channel in filter.iter() {
        Neuron::logger().debug(|| format!("filter size = {}x{}", channel.rows(), channel.cols()));
        weigths.push(Box::new(channel.clone()));
      }
      
      final_weigths.push(Weigths {
        name: format!("{}_{}", self.name, idx),
        weights: weigths,
        bias: vec![Box::new(Tensor::from_data(1, 1, vec![self.bias[idx]]))]
      });
    }

    final_weigths
  }

  fn set_weights(&mut self, weights: Vec<Weigths>) {
    for w in weights {
      if w.name.starts_with(&self.name) {
        for (f_idx, filter )in self.filters.iter_mut().enumerate() {
          if w.name == format!("{}_{}", self.name, f_idx) {
            for (c_idx,channel )in filter.iter_mut().enumerate() {
              *channel = *w.weights[c_idx].clone();
            }
            self.bias[f_idx] = w.bias[0].data()[0];
          }
        }
      }
    }
  }
}

impl Propagation for ConvLayer {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {
    if let Some((forward_input, z1, ret)) = Neuron::executors().conv.forward(input, self.filters.clone(), self.filter_size, self.bias.clone(), &self.config){
      self.last_z1 = Some(z1);
      self.last_input = Some(forward_input);
      return Some(ret);
    }
    None
  }

  fn backward(&mut self, input: &Vec<Box<Tensor>>, first: bool) -> Option<Vec<Box<Tensor>>> {

    if let Some(ref last_z1) = self.last_z1 {
      if let Some(ref forward_input) = self.last_input {
        return Neuron::executors().conv.backward(input, forward_input, last_z1, &mut self.filters, self.filter_size, &mut self.bias, !first, &self.config);
      }
    }
    None
  }
  
  
  fn as_loader(&self) -> Option<&dyn Loader> {
    Some(self)
  }

  fn as_mut_loader(&mut self) -> Option<&mut dyn Loader> {
    Some(self)
  }

}

pub struct FlattenLayer {
  input_cols: usize,
  input_rows: usize,
  n_channels: usize
}

impl FlattenLayer {
  pub fn new() -> Self {
    FlattenLayer {
      input_rows: 0,
      input_cols: 0,
      n_channels: 0
    }
  }
}

impl Propagation for FlattenLayer {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {

    let timer = Instant::now();

    self.input_rows = input[0].rows();
    self.input_cols = input[0].cols();
    self.n_channels = input.len();

    Neuron::logger().debug(|| format!("FlattenLayer Input Size (Forward) = {}x{}x{}",input[0].rows(), input[0].cols(), input.len()));
    Neuron::logger().debug(|| format!("FlattenLayer Input (Forward) = {:?}",input));

    let mut tmp = Vec::new();
    for i in input.iter() {
      for j in 0..i.rows() {
        for k in 0..i.cols() {
          tmp.push(i.get(j,k));
        }
      }
    }

    let t = Tensor::from_data(tmp.len(), 1, tmp);

    Neuron::logger().debug(|| format!("FlattenLayer Output Size (Forward) = {}x{}",t.rows(), t.cols()));
    Neuron::logger().debug(|| format!("FlattenLayer Output (Forward) = {:?}",t));

    Neuron::logger().profiling(|| format!("FlattenLayer Forward Time = {}ms", timer.elapsed().as_millis()));

    Some(vec![Box::new(t)])
  }

  fn backward(&mut self, input: &Vec<Box<Tensor>>, _: bool) -> Option<Vec<Box<Tensor>>> {

    let timer = Instant::now();

    let mut output = Vec::new();

    Neuron::logger().debug(|| format!("FlattenLayer Input Size (Backward) = {}x{}x{}",input[0].rows(), input[0].cols(), input.len()));

    for n_channel in 0..self.n_channels {
      let mut tmp = Tensor::new(self.input_rows, self.input_cols);
      for i in 0..self.input_rows {
        for j in 0..self.input_cols {
          tmp.set(i,j, input[0].data()[(n_channel * (self.input_cols*self.input_rows)) + (i * self.input_cols + j)])
        }
      }
      output.push(Box::new(tmp));
    }

    Neuron::logger().debug(|| format!("FlattenLayer Output Size (Backward) = {}x{}x{}",output[0].rows(), output[0].cols(),output.len()));
    Neuron::logger().debug(|| format!("FlattenLayer Output (Backward) = {:?}",output));

    Neuron::logger().profiling(|| format!("FlattenLayer Backward Time = {}ms", timer.elapsed().as_millis()));

    Some(output)
  }

  fn as_loader(&self) -> Option<&dyn Loader> {
    None
  }

  fn as_mut_loader(&mut self) -> Option<&mut dyn Loader> {
    None  
  }

}

pub struct PoolingLayerConfig {
  pub stride: usize
}

pub trait PoolingLayerExecutor {
  fn forward(&self, input: &Vec<Box<Tensor>>, filter_size: (usize, usize), config: &PoolingLayerConfig) -> Option<(Vec<Box<Tensor>>, Vec<Box<Tensor>>)>;
  fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, filter_size: (usize, usize), activate: bool, config: &PoolingLayerConfig) -> Option<Vec<Box<Tensor>>>;
}

pub struct PoolingLayer {
  config: PoolingLayerConfig,
  filter_size: (usize,usize),

  last_input: Option<Vec<Box<Tensor>>>,
}

impl PoolingLayer {
  pub fn new(filter_size: (usize, usize), config: PoolingLayerConfig) -> Self {
    PoolingLayer {
      config,
      filter_size,
      last_input: None
    }
  }
}

impl Propagation for PoolingLayer {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {    
    if let Some((forward_input,ret)) = Neuron::executors().pooling.forward(input, self.filter_size, &self.config) {
      self.last_input = Some(forward_input.clone());
      return Some(ret);
    }
    None
  }

  fn backward(&mut self, input: &Vec<Box<Tensor>>, first: bool) -> Option<Vec<Box<Tensor>>> {
    if let Some(ref forward_input) = self.last_input {
      return Neuron::executors().pooling.backward(input, forward_input,  self.filter_size, !first, &self.config);
    }
    None
  }
  
  fn as_loader(&self) -> Option<&dyn Loader> {
    None
  }

  fn as_mut_loader(&mut self) -> Option<&mut dyn Loader> {
    None  
  }


}