use crate::math::Tensor;

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
    let data:Vec<f64> = value.data().iter().map(|value| {if *value > 0.0 { *value } else { 0.0 }} ).collect();
    Tensor::from_data(value.rows(), value.cols(), data)
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
}

impl Activation for Sigmoid {
  fn forward(&self, value: &Tensor) -> Tensor {
    println!("Sigmoid for: {}", value);
    let data:Vec<f64> = value.data().iter().map(|value| 1.0 / (1.0 + (-value).exp()) ).collect();
    println!("Result: {}", Tensor::from_data(value.rows(), value.cols(), data));
    Tensor::from_data(value.rows(), value.cols(), data)
  }

  fn backward(&self, value: &Tensor) -> Tensor {
    let data:Vec<f64> = value.data().iter().map(|value| value * (1.0 - value) ).collect();
    Tensor::from_data(value.rows(), value.cols(), data)
  }
}