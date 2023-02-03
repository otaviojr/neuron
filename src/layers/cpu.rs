use std::time::Instant;

use crate::{math::Tensor, Neuron};

use super::{DenseLayerExecutor, DenseLayerConfig, ConvLayerExecutor, ConvLayerConfig, PoolingLayerExecutor, PoolingLayerConfig, ConvBatchNormalizationLayerExecutor, ConvBatchNormalizationLayerConfig};

pub struct DenseLayerCPU;

impl DenseLayerCPU {
  pub fn init() -> Self {
    DenseLayerCPU
  }
}

impl DenseLayerExecutor for DenseLayerCPU {
    fn forward(&self, input: &Vec<Box<Tensor>>, weights: &Tensor, bias: &Tensor, config: &DenseLayerConfig) -> Option<(Vec<Box<Tensor>>, Tensor, Vec<Box<Tensor>>)> {

      let timer = Instant::now();

      let input = &input[0];

      Neuron::logger().debug(|| format!("Bias = {:?}", bias));
      Neuron::logger().debug(|| format!("Layer weights size = {}x{}", weights.rows(), weights.cols()));
      Neuron::logger().debug(|| format!("Layer weights = {:?}", weights));
      Neuron::logger().debug(|| format!("DenseLayer Input (Forward) = {:?}", input));

      let mut z1_1 = weights.mul(&input).unwrap();

      Neuron::logger().profiling(|| format!("DenseLayer Forward Time (weights mul) = {}ns", timer.elapsed().as_millis()));

      Neuron::logger().debug(|| format!("z1_1 = {}x{}", z1_1.rows(), z1_1.cols()));
      Neuron::logger().debug(|| format!("z1_1 = {:?}", z1_1));

      let b_bias = bias.broadcast(z1_1.rows(), z1_1.cols()).unwrap();

      Neuron::logger().debug(|| format!("b_bias = {}x{}", b_bias.rows(), b_bias.cols()));
      Neuron::logger().debug(|| format!("b_bias = {:?}", b_bias));

      let z1 = z1_1.add(&b_bias).unwrap();  

      Neuron::logger().profiling(|| format!("DenseLayer Forward Time (add bias) = {}ns", timer.elapsed().as_millis()));

      let last_z1 = z1.clone();
      let last_input = vec![input.clone()];

      Neuron::logger().profiling(|| format!("DenseLayer Forward Time (clone) = {}ns", timer.elapsed().as_millis()));

      let ret = vec![Box::new(config.activation.forward(&z1).unwrap())];
  
      Neuron::logger().profiling(|| format!("DenseLayer Forward Time (activation) = {}ns", timer.elapsed().as_millis()));

      Neuron::logger().debug(|| format!("DenseLayer Output Before Activation(Forward) = {:?}", z1));
      Neuron::logger().debug(|| format!("DenseLayer Output (Forward) = {:?}", ret));
  
      Neuron::logger().profiling(|| format!("DenseLayer Forward Time = {}ns", timer.elapsed().as_millis()));
      Some((last_input, last_z1, ret))
    }

    fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, last_z1: &Tensor, weights: &mut Tensor, bias: &mut Tensor, activate: bool, config: &DenseLayerConfig) -> Option<Vec<Box<Tensor>>> {

      let timer = Instant::now();

      Neuron::logger().debug(|| format!("DenseLayer input size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));
      Neuron::logger().debug(|| format!("DenseLayer input (Backward) = {:?}", input));
  
      let input = &input[0];
  
      let dz;
      if activate {
        dz = input.clone().as_mut().mul_wise(&config.activation.backward(&last_z1).unwrap()).unwrap();
      } else {
        dz = Tensor::from_data(input.rows(), input.cols(), input.data().to_owned());
      }

      Neuron::logger().debug(|| format!("dz size = {}x{}", dz.rows(), dz.cols()));

      let mut fi = forward_input[0].clone();
      let mut forward_input = fi.as_mut();

      Neuron::logger().debug(|| format!("forward_input size = {}x{}", forward_input.rows(), forward_input.cols()));

      let mut dw = dz.mul(&forward_input.transpose().unwrap()).unwrap().div_value(forward_input.cols() as f32).unwrap();

      Neuron::logger().debug(|| format!("dw size = {}x{}", dw.rows(), dw.cols()));

      let mut db = Tensor::from_data(dz.rows(), dz.cols(), dz.data().to_owned());
      db = db.sum_row().unwrap().div_value(forward_input.cols() as f32).unwrap();

      Neuron::logger().debug(|| format!("db size = {}x{}", db.rows(), db.cols()));

      let zl = vec![Box::new(weights.transpose().unwrap().mul(&dz).unwrap())];

      Neuron::logger().debug(|| format!("DenseLayer output size (Backward) = {}x{}x{}", zl[0].rows(), zl[0].cols(), zl.len()));
      Neuron::logger().debug(|| format!("DenseLayer output (Backward) = {:?}", zl));

      let ret = Some(zl);

      Neuron::logger().debug(|| format!("weights size = {}x{}", weights.rows(), weights.cols()));

      *weights = weights.sub(&dw.mul_value(config.learn_rate).unwrap()).unwrap();
      *bias = bias.sub(&db.mul_value(config.learn_rate).unwrap()).unwrap();

      Neuron::logger().profiling(|| format!("DenseLayer Backward Time = {}ns", timer.elapsed().as_millis()));
  
      return ret;
    }
}

pub struct ConvLayerCPU;

impl ConvLayerCPU {
  pub fn init() -> Self {
    ConvLayerCPU
  }
}

impl ConvLayerExecutor for ConvLayerCPU {
  fn forward(&self, input: &Vec<Box<Tensor>>, filters: Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: Vec<f32>, config: &ConvLayerConfig) -> Option<(Vec<Box<Tensor>>, Vec<Box<Tensor>>, Vec<Box<Tensor>>)> {

    let timer = Instant::now();

    let result_height = (((input[0].rows() as f32 + 2.0* config.padding as f32 - filter_size.0 as f32)/config.stride as f32) + 1.0).floor() as usize;
    let result_width = (((input[0].cols() as f32 + 2.0* config.padding as f32 - filter_size.1 as f32)/config.stride as f32) + 1.0).floor() as usize;
    let mut result_final = Vec::new();
    let mut z1_final = Vec::new();

    Neuron::logger().debug(|| format!("ConvLayer input size (Forward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));
    Neuron::logger().debug(|| format!("ConvLayer input (Forward) = {:?}", input));

    for (f,b) in filters.iter().zip(bias.iter()) {
      let mut result_channels = Vec::new();
      let mut z1_channels = Vec::new();
      for (inp,fc) in input.iter().zip(f.iter()) {
        let mut z1 = Tensor::new(result_height, result_width);

        for result_y in 0..result_height {
          for result_x in 0..result_width {

            let i = (result_y * config.stride) as isize;
            let j = (result_x * config.stride) as isize;

            let mut sum = *b;
            for k in -(config.padding as isize)..(filter_size.0+config.padding) as isize {
              for l in -(config.padding as isize)..(filter_size.1+config.padding) as isize {
                if i+k >= 0 && i+k < inp.rows() as isize && j+l >= 0 && j+l < inp.cols() as isize{
                  sum += inp.get((i+k) as usize,(j+l) as usize) * fc.get(k as usize+config.padding,l as usize+config.padding);
                }
              }
            }
            z1.set(result_y, result_x, sum);
          }
        }
        let result = config.activation.forward(&z1).unwrap();
        result_channels.push(result);
        z1_channels.push(Box::new(z1));
      }
      result_final.push(result_channels); 
      z1_final.push(z1_channels); 
    }

    let mut output = Vec::new();
    let mut z1 = Vec::new();
    for (i,z) in result_final.iter().zip(z1_final.iter()) {
      let final_result = i.iter()
                                .fold(Some(Tensor::new(result_height, result_width).zero().unwrap()), |a,b| Some(a.unwrap().add(b).unwrap()))
                                .unwrap();

      output.push(Box::new(final_result));

      let final_z1 = z.iter()
                                .fold(Some(Tensor::new(z[0].rows(), z[0].cols()).zero().unwrap()), |a,b| Some(a.unwrap().add(b).unwrap()))
                                .unwrap();

      z1.push(Box::new(final_z1))
    }

    let last_input = input.clone();
    let last_z1 = z1.clone();

    Neuron::logger().debug(|| format!("ConvLayer Filter Size (Forward) = {}x{}x{}", filters[0][0].rows(), filters[0][0].cols(), filters[0].len()));
    Neuron::logger().debug(|| format!("ConvLayer Filter (Forward) = {:?}", filters));
    Neuron::logger().debug(|| format!("ConvLayer Output Size (Forward) = {}x{}x{}", output[0].rows(), output[0].cols(), output.len()));
    Neuron::logger().debug(|| format!("ConvLayer Output (Forward) = {:?}", output));

    Neuron::logger().profiling(|| format!("ConvLayer Forward Time = {}ns", timer.elapsed().as_millis()));

    Some((last_input, last_z1, output))
  }

  fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, last_z1: &Vec<Box<Tensor>>, filters: &mut Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: &mut Vec<f32>, _: bool, config: &ConvLayerConfig) -> Option<Vec<Box<Tensor>>> {
 
    let timer = Instant::now();
 
    let mut final_output = Vec::new();
    let mut final_dw = Vec::new();
    let mut final_db= Vec::new();
    
    Neuron::logger().debug(|| format!("ConvLayer Input (Backward) = {:?}", input));
    Neuron::logger().debug(|| format!("ConvLayer Input Size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));

    for (((f,inp), b),z1) in filters.iter_mut().zip(input.iter()).zip(bias.iter()).zip(last_z1.iter()) {
      let mut dw_channel = Vec::new();
      //let row_pad = (forward_input[0].rows() - inp.rows())/2;
      //let col_pad = (forward_input[0].cols() - inp.cols())/2;
      //let mut output = inp.pad(row_pad, col_pad).unwrap();
      let mut output = Tensor::new(forward_input[0].rows(), forward_input[0].cols()).zero().unwrap();
      let mut db = 0.0;
      for (fi,fc) in forward_input.iter().zip(f.iter_mut()) {

        let dz = inp.clone().as_mut().mul_wise(&config.activation.backward(&z1).unwrap()).unwrap();
        let mut dw = Tensor::new(fc.rows(), fc.cols()).zero().unwrap();

        for i in (0..fi.rows()-filter_size.0).step_by(config.stride) {
          for j in (0 .. fi.cols()-filter_size.1).step_by(config.stride) {
            for k in 0 .. filter_size.0 {
              for l in 0 .. filter_size.1 {
                output.set(i/config.stride,j/config.stride,output.get(i/config.stride,j/config.stride) + (dz.get(i/config.stride,j/config.stride) * fc.get(k,l)));
                dw.set(k,l,dw.get(k,l) + fi.get(i+k, j+l) * dz.get(i/config.stride,j/config.stride));
              }
            }
            db += dz.get(i/config.stride,j/config.stride);
          }
        }
        dw_channel.push(dw);
      }
      final_output.push(Box::new(output));
      final_db.push(db);
      final_dw.push(dw_channel);
    }

    Neuron::logger().debug(|| format!("CNN final_dw (Backward) = {:?}", final_dw));
    Neuron::logger().debug(|| format!("CNN final_db (Backward) = {:?}", final_db));

    for (((f,dw),b),db) in filters.iter_mut().zip(final_dw.iter()).zip(bias.iter_mut()).zip(final_db.iter()) {
      for (fc,dw_channel) in f.iter_mut().zip(dw.iter()) {
        for k in 0.. fc.rows() {
          for l in 0.. fc.cols() {
            fc.set(k,l,fc.get(k,l) - (dw_channel.get(k,l) * config.learn_rate));
            *b = *b - (db * config.learn_rate);
          }
        }
      }
    }

    Neuron::logger().debug(|| format!("CNN Filters (Backward) = {:?}", filters));
    Neuron::logger().debug(|| format!("CNN Output (Backward) = {:?}", final_output));
    Neuron::logger().debug(|| format!("CNN Output size (Backward) = {}x{}x{}", final_output[0].rows(), final_output[0].cols(), final_output.len()));

    Neuron::logger().profiling(|| format!("ConvLayer Backward Time = {}ns", timer.elapsed().as_nanos()));

    Some(final_output)  
  }
}

pub struct PoolingLayerCPU;

impl PoolingLayerCPU {
  pub fn new() -> PoolingLayerCPU {
    PoolingLayerCPU
  }
}

impl PoolingLayerCPU {
  pub fn init() -> PoolingLayerCPU {
    PoolingLayerCPU
  }
}

impl PoolingLayerExecutor for PoolingLayerCPU {
  fn forward(&self, input: &Vec<Box<Tensor>>, filter_size: (usize, usize), config: &PoolingLayerConfig) -> Option<(Vec<Box<Tensor>>, Vec<Box<Tensor>>)> {

    
    let timer = Instant::now();

    let result_height = (((input[0].rows() as f32 - filter_size.0 as f32)/config.stride as f32) + 1.0).floor() as usize;
    let result_width = (((input[0].cols() as f32 - filter_size.1 as f32)/config.stride as f32) + 1.0).floor() as usize;
    
    Neuron::logger().debug(|| format!("PoolingLayer Input (Forward) = {:?}", input));
    Neuron::logger().debug(|| format!("PoolingLayer Input size (Forward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));
    Neuron::logger().debug(|| format!("PoolingLayer Output size (Forward) = {}x{}x{}", result_height, result_width, input.len()));

    let mut result_final = Vec::new();
    for inp in input.iter() {
      let mut result = Tensor::new(result_height, result_width);

      for result_y in 0 .. result_height {
        for result_x in 0 .. result_width {

          let i = result_y * config.stride;
          let j = result_x * config.stride;

          let mut max = std::f32::NEG_INFINITY;
          for k in 0 .. filter_size.0 {
            for l in 0 .. filter_size.1 {
              let value = inp.get(i+k,j+l);
              if  value > max {
                max = value;
              }
            }
          }
          result.set(result_y, result_x, max);
        }
      }
      result_final.push(Box::new(result));
    }

    let last_input = input.clone();

    Neuron::logger().debug(|| format!("PoolingLayer Output (Forward) = {:?}", result_final));
    Neuron::logger().profiling(|| format!("PoolingLayer Forward Time = {}ns", timer.elapsed().as_nanos()));

    Some((last_input,result_final))
  }

  fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, filter_size: (usize, usize), activate: bool, config: &PoolingLayerConfig) -> Option<Vec<Box<Tensor>>> {
    let timer = Instant::now();

    let mut result_final = Vec::new();

    Neuron::logger().debug(|| format!("PoolingLayer Input (Backward) = {:?}", input));
    Neuron::logger().debug(|| format!("PoolingLayer Input size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));

    for (inp,fi) in input.iter().zip(forward_input.iter()) {
      let mut result =  Tensor::new(fi.rows(), fi.cols());
      for i in (0 .. fi.rows()-filter_size.0).step_by(config.stride) {
        for j in (0 .. fi.cols()-filter_size.1).step_by(config.stride) {
          let mut max = 0.0;
          let mut max_k = 0;
          let mut max_l = 0;
          for k in 0 .. filter_size.0 {
            for l in 0 .. filter_size.1 {
              let value = fi.get(i+k,j+l);
              if value > max {
                max = value;
                max_k = k;
                max_l = l;
              }
            }
          }
          result.set(i+max_k,j+max_l, inp.get(i/config.stride,j/config.stride));
        }
      }
      result_final.push(Box::new(result));  
    }

    Neuron::logger().debug(|| format!("PoolingLayer Output (Backward) = {:?}", result_final));
    Neuron::logger().debug(|| format!("PoolingLayer Output size (Backward) = {}x{}x{}", result_final[0].rows(), result_final[0].cols(), result_final.len()));

    Neuron::logger().profiling(|| format!("PoolingLayer Backward Time = {}ns", timer.elapsed().as_nanos()));
    
    Some(result_final)
  }
}

pub struct ConvBatchNormalizationLayerCPU;

impl ConvBatchNormalizationLayerCPU {
  pub fn init() -> Self {
    ConvBatchNormalizationLayerCPU
  }
}

impl ConvBatchNormalizationLayerExecutor for ConvBatchNormalizationLayerCPU {
  fn forward(&self, input: &Vec<Box<Tensor>>, beta: &Vec<Box<Tensor>>, gamma: &Vec<Box<Tensor>>, config: &ConvBatchNormalizationLayerConfig) -> Option<(Vec<Box<Tensor>>, Vec<Box<Tensor>>, Vec<Box<Tensor>>, Vec<Box<Tensor>>)> {

    let timer = Instant::now();

    Neuron::logger().debug(|| format!("ConvBatchNormalizationLayer Input (Forward) = {:?}", input));
    Neuron::logger().debug(|| format!("ConvBatchNormalizationLayer Input size (Forward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));

    let batch_size = input[0].data().len() as f32;

    let mut mean = Vec::new();
    let mut var = Vec::new();
    let mut x_hat = Vec::new();
    let mut result = Vec::new();

    for (idx,inp) in input.iter().enumerate() {
      let f_sum:f32 = inp.data().iter().sum();
      let f_mean = f_sum / batch_size;

      let f_var: f32 = inp.data().iter().map(|x| (x - f_mean).powi(2)).sum::<f32>();

      let sub_x_hat: Vec<f32> = inp.data().iter().map(|x| (*x - f_mean) / (f_var + config.epsilon).sqrt()).collect();

      let f_result = sub_x_hat.iter().map(|x| x * gamma[0].get(idx,0) + beta[0].get(idx, 0)).collect();
      let result_tensor = Tensor::from_data(inp.rows(), inp.cols(), f_result);

      mean.push(f_mean);
      var.push(f_var);
      let x_hat_tensor = Tensor::from_data(inp.rows(), inp.cols(), sub_x_hat);
      x_hat.push(Box::new(x_hat_tensor));
      result.push(Box::new(result_tensor));
    }

    Neuron::logger().debug(|| format!("ConvBatchNormalizationLayer Output (Forward) = {:?}", result));
    Neuron::logger().profiling(|| format!("ConvBatchNormalizationLayer Forward Time = {}ns", timer.elapsed().as_nanos()));

    let var = vec![Box::new(Tensor::from_data(input.len(),1,var))];
    let mean = vec![Box::new(Tensor::from_data(input.len(),1,mean))];

    Some((result, x_hat, var, mean))
  }

  fn backward(&self, input: &Vec<Box<Tensor>>, beta: &mut Vec<Box<Tensor>>, gamma: &mut Vec<Box<Tensor>>, input_x_hat: &Vec<Box<Tensor>>, var: &Vec<Box<Tensor>>, mean: &Vec<Box<Tensor>>, config: &ConvBatchNormalizationLayerConfig) -> Option<Vec<Box<Tensor>>> {
    let timer = Instant::now();

    Neuron::logger().debug(|| format!("ConvBatchNormalizationLayer Input (Backward) = {:?}", input));
    Neuron::logger().debug(|| format!("ConvBatchNormalizationLayer Input size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));

    let batch_size = input[0].data().len() as f32;
    let mut dx_hat = Vec::new();

    for ((idx, inp), x_hat) in input.iter().enumerate().zip(input_x_hat.iter()) {

      let f_mean:f32 = mean[0].get(idx,0);
      let f_var: f32 = var[0].get(idx, 0);
      let f_std = (f_var + config.epsilon).sqrt();
      let ivar = 1.0 / f_std;

      let f_d_beta:f32 = inp.data().iter().sum::<f32>() / batch_size;
      let f_d_gamma: f32 = x_hat.clone().mul(inp).unwrap().data().iter().sum::<f32>() / batch_size;

      let mut dx = *inp.clone();
      dx = dx.mul_value(gamma[0].get(idx,0)).unwrap();
      let divar = dx.data().iter().zip(x_hat.data().iter()).map(|(x,x_hat)| (x - f_mean) * x_hat).sum::<f32>() / batch_size;
      dx.mut_data().iter_mut().for_each(|dx| *dx = *dx * ivar);

      let dsqrtvar = - divar / f_std.powi(2);
      let dvar = 0.5 * (dsqrtvar / f_std);

      let dsq = (1.0 + batch_size) * dvar / (batch_size * batch_size);
      dx.mut_data().iter_mut().zip(x_hat.data().iter()).for_each(|(dx,x)| *dx = *dx + 2.0*(x-f_mean)*dsq);

      let dmu = dx.data().iter().sum::<f32>() * -1.0 / batch_size;

      dx.mut_data().iter_mut().for_each(|x| *x = *x + dmu);
        
      let new_gamma = gamma[0].get(idx,0) - config.learn_rate * f_d_gamma;
      let new_beta = beta[0].get(idx,0) - config.learn_rate * f_d_beta;
        
      gamma[0].set(idx,0, new_gamma);
      beta[0].set(idx,0, new_beta);

      dx_hat.push(Box::new(dx));
    }

    Neuron::logger().debug(|| format!("ConvBatchNormalizationLayer Output (Backward) = {:?}", dx_hat));
    Neuron::logger().debug(|| format!("ConvBatchNormalizationLayer Output size (Backward) = {}x{}x{}", dx_hat[0].rows(), dx_hat[0].cols(), dx_hat.len()));

    Neuron::logger().profiling(|| format!("ConvBatchNormalizationLayer Backward Time = {}ns", timer.elapsed().as_nanos()));

    Some(dx_hat)
  }
}