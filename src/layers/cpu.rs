use std::time::Instant;

use crate::{math::Tensor, Neuron, Propagation};

use super::{DenseLayerExecutor, DenseLayerConfig, ConvLayerExecutor, ConvLayerConfig, PoolingLayerExecutor, PoolingLayerConfig};

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

      let z1_1 = weights.mul(&input).unwrap();

      Neuron::logger().profiling(|| format!("DenseLayer Forward Time (weights mul) = {}ms", timer.elapsed().as_millis()));

      Neuron::logger().debug(|| format!("z1_1 = {}x{}", z1_1.rows(), z1_1.cols()));
      Neuron::logger().debug(|| format!("z1_1 = {:?}", z1_1));

      let b_bias = bias.broadcast(z1_1.rows(), z1_1.cols()).unwrap();

      Neuron::logger().debug(|| format!("b_bias = {}x{}", b_bias.rows(), b_bias.cols()));
      Neuron::logger().debug(|| format!("b_bias = {:?}", b_bias));

      let z1 = z1_1.add(&b_bias).unwrap();  

      Neuron::logger().profiling(|| format!("DenseLayer Forward Time (add bias) = {}ms", timer.elapsed().as_millis()));

      let last_z1 = z1.clone();
      let last_input = vec![input.clone()];

      Neuron::logger().profiling(|| format!("DenseLayer Forward Time (clone) = {}ms", timer.elapsed().as_millis()));

      let ret = vec![Box::new(config.activation.forward(&z1).unwrap())];
  
      Neuron::logger().profiling(|| format!("DenseLayer Forward Time (activation) = {}ms", timer.elapsed().as_millis()));

      Neuron::logger().debug(|| format!("DenseLayer Output Before Activation(Forward) = {:?}", z1));
      Neuron::logger().debug(|| format!("DenseLayer Output (Forward) = {:?}", ret));
  
      Neuron::logger().profiling(|| format!("DenseLayer Forward Time = {}ms", timer.elapsed().as_millis()));
      Some((last_input, last_z1, ret))
    }

    fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, last_z1: &Tensor, weights: &mut Tensor, bias: &mut Tensor, activate: bool, config: &DenseLayerConfig) -> Option<Vec<Box<Tensor>>> {

      let timer = Instant::now();

      Neuron::logger().debug(|| format!("DenseLayer input size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));
      Neuron::logger().debug(|| format!("DenseLayer input (Backward) = {:?}", input));
  
      let input = &input[0];
  
      let dz;
      if activate {
        dz = input.mul_wise(&config.activation.backward(&last_z1).unwrap()).unwrap();
      } else {
        dz = Tensor::from_data(input.rows(), input.cols(), input.data().to_owned());
      }

      Neuron::logger().debug(|| format!("dz size = {}x{}", dz.rows(), dz.cols()));

      let forward_input = &forward_input[0];

      Neuron::logger().debug(|| format!("forward_input size = {}x{}", forward_input.rows(), forward_input.cols()));

      let dw = dz.mul(&forward_input.transpose().unwrap()).unwrap().div_value(forward_input.cols() as f64).unwrap();

      Neuron::logger().debug(|| format!("dw size = {}x{}", dw.rows(), dw.cols()));

      let mut db = Tensor::from_data(dz.rows(), dz.cols(), dz.data().to_owned());
      db = db.sum_row().unwrap().div_value(forward_input.cols() as f64).unwrap();

      Neuron::logger().debug(|| format!("db size = {}x{}", db.rows(), db.cols()));

      let zl = vec![Box::new(weights.transpose().unwrap().mul(&dz).unwrap())];

      Neuron::logger().debug(|| format!("DenseLayer output size (Backward) = {}x{}x{}", zl[0].rows(), zl[0].cols(), zl.len()));
      Neuron::logger().debug(|| format!("DenseLayer output (Backward) = {:?}", zl));

      let ret = Some(zl);

      Neuron::logger().debug(|| format!("weights size = {}x{}", weights.rows(), weights.cols()));

      *weights = weights.sub(&dw.mul_value(config.learn_rate).unwrap()).unwrap();
      *bias = bias.sub(&db.mul_value(config.learn_rate).unwrap()).unwrap();

      Neuron::logger().profiling(|| format!("DenseLayer Backward Time = {}ms", timer.elapsed().as_millis()));
  
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
  fn forward(&self, input: &Vec<Box<Tensor>>, filters: Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: Vec<f64>, config: &ConvLayerConfig) -> Option<(Vec<Box<Tensor>>, Vec<Box<Tensor>>, Vec<Box<Tensor>>)> {

    let timer = Instant::now();

    let result_height = (((input[0].rows() as f64 + 2.0* config.padding as f64 - filter_size.0 as f64)/config.stride as f64) + 1.0).floor() as usize;
    let result_width = (((input[0].cols() as f64 + 2.0* config.padding as f64 - filter_size.1 as f64)/config.stride as f64) + 1.0).floor() as usize;
    let mut result_final = Vec::new();
    let mut z1_final = Vec::new();

    Neuron::logger().debug(|| format!("ConvLayer input size (Forward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));
    Neuron::logger().debug(|| format!("ConvLayer input (Forward) = {:?}", input));

    for (f,b) in filters.iter().zip(bias.iter()) {
      let mut result_channels = Vec::new();
      let mut z1_channels = Vec::new();
      for (inp,fc) in input.iter().zip(f.iter()) {
        let mut result = Tensor::zeros(result_height, result_width);
        for i in (0..inp.rows() - filter_size.0).step_by(config.stride) {
          for j in (0..inp.cols() - filter_size.1).step_by(config.stride) {
            let mut sum = *b;
            for k in 0 .. filter_size.0 {
              for l in 0 .. filter_size.1 {
                sum += inp.get(i+k,j+l) * fc.get(k,l);
              }
            }
            result.set(i/config.stride, j/config.stride, sum);
          }
        }
        let z1 = config.activation.forward(&result).unwrap();
        result_channels.push(z1.clone());
        z1_channels.push(Box::new(result));
      }
      result_final.push(result_channels); 
      z1_final.push(z1_channels); 
    }

    let mut output = Vec::new();
    let mut z1 = Vec::new();
    for (i,z) in result_final.iter().zip(z1_final.iter()) {
      let final_result = i.iter()
                                .fold(Some(Tensor::zeros(result_height, result_width)), |a,b| Some(a.unwrap().add(b).unwrap()))
                                .unwrap_or(Tensor::zeros(result_height, result_width));

      output.push(Box::new(final_result));

      let final_z1 = z.iter()
                                .fold(Some(Tensor::zeros(z[0].rows(), z[0].cols())), |a,b| Some(a.unwrap().add(b).unwrap()))
                                .unwrap_or(Tensor::zeros(z[0].rows(), z[0].cols()));

      z1.push(Box::new(final_z1))
    }

    let last_input = input.clone();
    let last_z1 = z1.clone();

    Neuron::logger().debug(|| format!("ConvLayer Filter Size (Forward) = {}x{}x{}", filters[0][0].rows(), filters[0][0].cols(), filters[0].len()));
    Neuron::logger().debug(|| format!("ConvLayer Filter (Forward) = {:?}", filters));
    Neuron::logger().debug(|| format!("ConvLayer Output Size (Forward) = {}x{}x{}", output[0].rows(), output[0].cols(), output.len()));
    Neuron::logger().debug(|| format!("ConvLayer Output (Forward) = {:?}", output));

    Neuron::logger().profiling(|| format!("ConvLayer Forward Time = {}ms", timer.elapsed().as_millis()));

    Some((last_input, last_z1, output))
  }

  fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, last_z1: &Vec<Box<Tensor>>, filters: &mut Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: &mut Vec<f64>, _: bool, config: &ConvLayerConfig) -> Option<Vec<Box<Tensor>>> {
 
    let timer = Instant::now();
 
    let mut final_output = Vec::new();
    let mut final_dw = Vec::new();
    let mut final_db= Vec::new();
    
    Neuron::logger().debug(|| format!("ConvLayer Input (Backward) = {:?}", input));
    Neuron::logger().debug(|| format!("ConvLayer Input Size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));

    for (((f,inp), b),z1) in filters.iter_mut().zip(input.iter()).zip(bias.iter()).zip(last_z1.iter()) {
      let mut dw_channel = Vec::new();
      let row_pad = (forward_input[0].rows() - inp.rows())/2;
      let col_pad = (forward_input[0].cols() - inp.cols())/2;
      let mut output = inp.pad(row_pad, col_pad).unwrap();
      let mut db = 0.0;
      for (fi,fc) in forward_input.iter().zip(f.iter_mut()) {

        let dz = inp.mul_wise(&config.activation.backward(&z1).unwrap()).unwrap();
        let mut dw = Tensor::zeros(fc.rows(), fc.cols());

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
      final_output.push(Box::new(config.activation.backward(&output.add_value(*b).unwrap()).unwrap()));
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

    Neuron::logger().profiling(|| format!("ConvLayer Backward Time = {}ms", timer.elapsed().as_millis()));

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

    let result_height = (((input[0].rows() as f64 - filter_size.0 as f64)/config.stride as f64) + 1.0).floor() as usize;
    let result_width = (((input[0].cols() as f64 - filter_size.1 as f64)/config.stride as f64) + 1.0).floor() as usize;
    
    Neuron::logger().debug(|| format!("PoolingLayer Input (Forward) = {:?}", input));
    Neuron::logger().debug(|| format!("PoolingLayer Input size (Forward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));
    Neuron::logger().debug(|| format!("PoolingLayer Output size (Forward) = {}x{}x{}", result_height, result_width, input.len()));

    let mut result_final = Vec::new();
    for inp in input.iter() {
      let mut result = Tensor::zeros(result_height, result_width);
      for i in (0 .. inp.rows() - filter_size.0).step_by(config.stride) {
        for j in (0 .. inp.cols() - filter_size.1).step_by(config.stride) {
          let mut max = 0.0;
          for k in 0 .. filter_size.0 {
            for l in 0 .. filter_size.1 {
              let value = inp.get(i+k,j+l);
              if  value > max {
                max = value;
              }
            }
          }
          result.set(i/config.stride, j/config.stride, max);
        }
      }
      result_final.push(Box::new(result));
    }

    let last_input = input.clone();

    Neuron::logger().debug(|| format!("PoolingLayer Output (Forward) = {:?}", result_final));
    Neuron::logger().profiling(|| format!("PoolingLayer Forward Time = {}ms", timer.elapsed().as_millis()));

    Some((last_input,result_final))
  }

  fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, filter_size: (usize, usize), activate: bool, config: &PoolingLayerConfig) -> Option<Vec<Box<Tensor>>> {
    let timer = Instant::now();

    let mut result_final = Vec::new();

    Neuron::logger().debug(|| format!("PoolingLayer Input (Backward) = {:?}", input));
    Neuron::logger().debug(|| format!("PoolingLayer Input size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));

    for (inp,fi) in input.iter().zip(forward_input.iter()) {
      let mut result =  Tensor::zeros(fi.rows(), fi.cols());
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

    Neuron::logger().profiling(|| format!("PoolingLayer Backward Time = {}ms", timer.elapsed().as_millis()));
    
    Some(result_final)
  }
}