use crate::math::Tensor;

pub struct Functions;

impl Functions{
  pub fn binary_cross_entropy_loss(predictions: &Tensor, targets: &Tensor) -> f32 {
    let mut loss = 0.0;

    for (prediction, target) in predictions.data().iter().zip(targets.data().iter()) {
        let v = -((target * prediction.ln()) + ((1.0 - target) * (1.0 - prediction).ln()));
        //println!("v={} for {} and {}",v, target, prediction);
        if !v.is_nan() {
          loss += v;
        }
    }

    -loss / targets.cols() as f32
  }
  
  pub fn binary_cross_entropy_loss_derivative(predictions: &Tensor, targets: &Tensor) -> Tensor {  
    let mut data = Vec::new();
    for (prediction, target) in predictions.data().iter().zip(targets.data().iter()) {
        data.push( -((target/prediction) - ((1.0-target)/(1.0-prediction))));
    }
    Tensor::from_data(targets.rows(), targets.cols(), data)
  }

  pub fn softmax_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {  
    let mut data = Vec::new();
    for (prediction, target) in predictions.data().iter().zip(targets.data().iter()) {
        data.push( -target * prediction.ln());
    }
    Tensor::from_data(targets.rows(), targets.cols(), data)
  }
}
