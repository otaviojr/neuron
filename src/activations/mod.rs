use crate::math::Tensor;
use std::f64::consts::E;

pub trait Activation {
  fn forward(&self, value: &Tensor) -> Tensor;
  fn backward(&self, value: &Tensor) -> Tensor;
}

pub struct ReLU {

}

impl ReLU {
  pub fn new() -> Self {
    ReLU {  }
  }
}

impl Activation for ReLU {
  fn forward(&self, value: &Tensor) -> Tensor {
    //println!("ReLU Entry: {}", value);
    let data:Vec<f64> = value.data().iter().map(|value| {if *value > 0.0 { *value } else { 0.0 }} ).collect();
    let ret = Tensor::from_data(value.rows(), value.cols(), data);
    //println!("ReLU Result: {}", ret);
    ret
  }

  fn backward(&self, value: &Tensor) -> Tensor {
    let data:Vec<f64> = value.data().iter().map(|value| {if *value > 0.0 { 1.0 } else { 0.0 }} ).collect();
    Tensor::from_data(value.rows(), value.cols(), data)
  }
}

pub struct Sigmoid {

}

impl Sigmoid {
  pub fn new() -> Self {
    Sigmoid {  }
  }

  fn sigmoid(value:f64) -> f64 {
    let ret = 1.0 / (1.0 + (-value).exp()); 
    if ret.is_nan() {0.0} else {ret}
  }
}

impl Activation for Sigmoid {
  fn forward(&self, value: &Tensor) -> Tensor {
    //println!("Sigmoid Entry: {}", value);
    let data:Vec<f64> = value.data().iter().map(|value| Sigmoid::sigmoid(*value) ).collect();
    let ret = Tensor::from_data(value.rows(), value.cols(), data);
    //println!("Sigmoid result: {}", ret);
    ret
  }

  fn backward(&self, value: &Tensor) -> Tensor {
    let data:Vec<f64> = value.data().iter().map(|value| Sigmoid::sigmoid(*value) * (1.0 - Sigmoid::sigmoid(*value)) ).collect();
    Tensor::from_data(value.rows(), value.cols(), data)
  }
}

pub struct Tanh {

}

impl Tanh {
  pub fn new() -> Self {
    Tanh {  }
  }

  fn tanh(value:f64) -> f64 {
    let ret = (value.exp() - (-value).exp()) / ( value.exp() + (-value).exp()); 
    if ret.is_nan() {0.0} else {ret}
  }
}

impl Activation for Tanh {
  fn forward(&self, value: &Tensor) -> Tensor {
    //println!("Sigmoid Entry: {}", value);
    let data:Vec<f64> = value.data().iter().map(|value| Tanh::tanh(*value) ).collect();
    let ret = Tensor::from_data(value.rows(), value.cols(), data);
    //println!("Sigmoid result: {}", ret);
    ret
  }

  fn backward(&self, value: &Tensor) -> Tensor {
    let data:Vec<f64> = value.data().iter().map(|value| 1.0 - Tanh::tanh(*value).powi(2) ).collect();
    Tensor::from_data(value.rows(), value.cols(), data)
  }
}