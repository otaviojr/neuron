use crate::math::Tensor;

pub struct Functions;

impl Functions{
  pub fn binary_cross_entropy_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let cost = Tensor::zeros(targets.rows(), targets.cols());
    
    //let mut loss = 0.0;
    //for (prediction, target) in predictions.iter().zip(targets.iter()) {
    //    loss += target * prediction.ln() + (1.0 - target) * (1.0 - prediction).ln();
    //}
    //-loss / predictions.len() as f32
  
    cost
  }
  
  pub fn binary_cross_entropy_loss_derivative(predictions: &Tensor, targets: &Tensor) -> Tensor {  
    let mut data = Vec::new();
    for (prediction, target) in predictions.data().iter().zip(targets.data().iter()) {
        data.push(prediction - target);
    }
    Tensor::from_data(targets.rows(), targets.cols(), data)
  }
}
