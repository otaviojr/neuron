pub trait Activation {
  fn forward(&self) -> f64;
  fn backward(&self) -> f64;
}

pub struct ReLU {

}

impl ReLU {
  pub fn new() -> Self {
    ReLU {  }
  }
}

impl Activation for ReLU {
  fn forward(&self) -> f64 {
    0.0
  }
  fn backward(&self) -> f64 {
    0.0
  }
}