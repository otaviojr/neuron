use crate::math::Tensor;

pub struct Functions;

impl Functions{
  pub fn binary_cross_entropy_loss(predictions: &Tensor, targets: &Tensor) -> f64 {
    let mut loss = 0.0;

    for (prediction, target) in predictions.data().iter().zip(targets.data().iter()) {
        loss += target * prediction.ln() + (1.0 - target) * (1.0 - prediction).ln();
    }
    -loss / predictions.data().len() as f64
  }
  
  pub fn binary_cross_entropy_loss_derivative(predictions: &Tensor, targets: &Tensor) -> Tensor {  
    let mut data = Vec::new();
    for (prediction, target) in predictions.data().iter().zip(targets.data().iter()) {
        data.push( ((1.0-target)/(1.0-prediction)) - (target/prediction));
    }
    Tensor::from_data(targets.rows(), targets.cols(), data)
  }
}
