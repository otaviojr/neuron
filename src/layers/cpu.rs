use crate::math::Tensor;

use super::{DenseLayerExecutor, DenseLayerConfig};

pub struct DenseLayerCPU;

impl DenseLayerCPU {
  pub fn init() -> Self {
    DenseLayerCPU
  }
}

impl DenseLayerExecutor for DenseLayerCPU {
    fn forward(&self, input: &Vec<Box<Tensor>>, weights: &Tensor, bias: &Tensor, config: &DenseLayerConfig) -> Option<(Vec<Box<Tensor>>, Tensor, Vec<Box<Tensor>>)> {
      let input = &input[0];
      println!("Bias = {:?}", bias);
      println!("Layer weights size = {}x{}", weights.rows(), weights.cols());
      println!("Layer weights = {:?}", weights);
      println!("DenseLayer Input (Forward) = {:?}", input);
      let z1_1 = weights.mul(&input);
      println!("z1_1 = {}x{}", z1_1.rows(), z1_1.cols());
      println!("z1_1 = {:?}", z1_1);
      let b_bias = bias.broadcast(z1_1.rows(), z1_1.cols());
      println!("b_bias = {}x{}", b_bias.rows(), b_bias.cols());
      println!("b_bias = {:?}", b_bias);
      let z1 = z1_1.add(&b_bias);
  
      let last_z1 = z1.clone();
      let last_input = vec![input.clone()];
      let ret = vec![Box::new(config.activation.forward(&z1))];
  
      println!("DenseLayer Output Before Activation(Forward) = {:?}", z1);
      println!("DenseLayer Output (Forward) = {:?}", ret);
  
      Some((last_input, last_z1, ret))
    }

    fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, last_z1: &Tensor, weights: &mut Tensor, bias: &mut Tensor, activate: bool, config: &DenseLayerConfig) -> Option<Vec<Box<Tensor>>> {
      println!("DenseLayer input size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len());
      println!("DenseLayer input (Backward) = {:?}", input);
  
      let input = &input[0];
  
      let dz;
      if activate {
        dz = input.mul_wise(&config.activation.backward(&last_z1));
      } else {
        dz = Tensor::from_data(input.rows(), input.cols(), input.data().to_owned());
      }
      println!("dz size = {}x{}", dz.rows(), dz.cols());
      //println!("dz = {}", dz);
      let forward_input = &forward_input[0];
      println!("forward_input size = {}x{}", forward_input.rows(), forward_input.cols());
      let dw = dz.mul(&forward_input.transpose()).div_value(forward_input.cols() as f64);
      println!("dw size = {}x{}", dw.rows(), dw.cols());
      //println!("dw = {}", dw);
      let mut db = Tensor::from_data(dz.rows(), dz.cols(), dz.data().to_owned());
      db = db.sum_row().div_value(forward_input.cols() as f64);
      println!("db size = {}x{}", db.rows(), db.cols());

      let zl = vec![Box::new(weights.transpose().mul(&dz))];
      println!("DenseLayer output size (Backward) = {}x{}x{}", zl[0].rows(), zl[0].cols(), zl.len());
      println!("DenseLayer output (Backward) = {:?}", zl);
      let ret = Some(zl);

      println!("weights size = {}x{}", weights.rows(), weights.cols());
      *weights = weights.sub(&dw.mul_value(config.learn_rate));
      *bias = bias.sub(&db.mul_value(config.learn_rate));
  
      return ret;
    }
}