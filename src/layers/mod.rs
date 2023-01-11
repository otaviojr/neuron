use std::result;

use rand::{Rng};
use rand_distr::{Uniform};

use crate::math::Tensor;
use crate::activations::Activation;

pub trait LayerPropagation {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>>;
  fn backward(&mut self, input: &Vec<Box<Tensor>>, first: bool) -> Option<Vec<Box<Tensor>>>;
}

pub struct LinearLayerConfig {
  pub activation: Box<dyn Activation>,
  pub learn_rate: f64,
  pub weights_low: f64,
  pub weights_high: f64

}

pub struct LinearLayer {
  config: LinearLayerConfig,
  weights: Tensor,
  bias: Tensor,

  last_input: Option<Vec<Box<Tensor>>>,
  last_z1: Option<Tensor>
}

impl LinearLayer {
  pub fn new(n_channels: usize, nodes: usize, config: LinearLayerConfig) -> Self {
    LinearLayer {
      weights: Tensor::randomHE(nodes,n_channels, nodes),
      bias: Tensor::randomHE(nodes,1, nodes),
      last_input: None,
      last_z1: None,
      config: config,
    }
  }
}

impl LayerPropagation for LinearLayer {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {
    let input = &input[0];
    println!("Bias = {:?}", self.bias);
    println!("Layer weights size = {}x{}", self.weights.rows(), self.weights.cols());
    println!("Layer weights = {:?}", self.weights);
    println!("LinearLayer Input (Forward) = {:?}", input);
    let z1_1 = self.weights.transpose().mul(&input);
    println!("z1_1 = {}x{}", z1_1.rows(), z1_1.cols());
    println!("z1_1 = {:?}", z1_1);
    let b_bias = self.bias.broadcast(z1_1.rows(), z1_1.cols());
    println!("b_bias = {}x{}", b_bias.rows(), b_bias.cols());
    println!("b_bias = {:?}", b_bias);
    let z1 = self.config.activation.forward(&z1_1.add(&b_bias));

    self.last_z1 = Some(z1.clone());
    self.last_input = Some(vec![input.clone()]);


    println!("LinearLayer Output Before Activation(Forward) = {:?}", z1);
    let ret = vec![Box::new(z1)];
    println!("LinearLayer Output (Forward) = {:?}", ret);

    Some(ret)
  }

  fn backward(&mut self, input: &Vec<Box<Tensor>>, first: bool) -> Option<Vec<Box<Tensor>>> {
    println!("LinearLayer input size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len());
    println!("LinearLayer input (Backward) = {:?}", input);

    let input = &input[0];

    if let Some(ref z1) = self.last_z1 {
      let dz;
      if first {
        dz = Tensor::from_data(input.rows(), input.cols(), input.data().to_owned());
      } else {
        dz = input.mul_wise(&self.config.activation.backward(z1));
      }
      println!("dz size = {}x{}", dz.rows(), dz.cols());
      //println!("dz = {}", dz);
      if let Some(ref forward_input) = self.last_input {
        let forward_input = &forward_input[0];
        println!("forward_input size = {}x{}", forward_input.rows(), forward_input.cols());
        let dw = dz.mul(&forward_input.transpose()).div_value(forward_input.cols() as f64);
        println!("dw size = {}x{}", dw.rows(), dw.cols());
        //println!("dw = {}", dw);
        let mut db = Tensor::from_data(dz.rows(), dz.cols(), dz.data().to_owned());
        db = db.sum_row().div_value(forward_input.cols() as f64);
        println!("db size = {}x{}", db.rows(), db.cols());

        let zl = vec![Box::new(self.weights.transpose().mul(&dz))];
        println!("LinearLayer output size (Backward) = {}x{}x{}", zl[0].rows(), zl[0].cols(), zl.len());
        println!("LinearLayer output (Backward) = {:?}", zl);
        let ret = Some(zl);

        println!("weights size = {}x{}", self.weights.rows(), self.weights.cols());
        self.weights = self.weights.sub(&dw.mul_value(self.config.learn_rate));
        self.bias = self.bias.sub(&db.mul_value(self.config.learn_rate));
    
        return ret;
      }
    }
    None
  }
}

pub struct ConvLayerConfig {
  pub activation: Box<dyn Activation>,
  pub learn_rate: f64,
  pub padding: usize,
  pub stride: usize,
  pub weights_low: f64,
  pub weights_high: f64
}

pub struct ConvLayer {
  config: ConvLayerConfig,
  filters: Vec<Vec<Tensor>>,
  filter_size: (usize,usize),
  bias: Vec<f64>,

  last_input: Option<Vec<Box<Tensor>>>,
  last_z1: Option<Vec<Box<Tensor>>>
}

impl ConvLayer {
  pub fn new(n_channels: usize, n_filters: usize, filter_size: (usize,usize), config: ConvLayerConfig) -> Self {
    let mut rng = rand::thread_rng();

    let mut filters = Vec::new();
    for i in 0 .. n_filters {
      let mut filter_channels = Vec::new();
      for j in 0 .. n_channels {
        let filter_channel = Tensor::randomHE(filter_size.0, filter_size.1, n_channels);
        filter_channels.push(filter_channel);
      }
      filters.push(filter_channels);
    }
  
    ConvLayer {
      filters,
      filter_size,
      bias: vec![0.0; n_filters],
      config,

      last_input: None,
      last_z1: None
    }
  }
}

impl LayerPropagation for ConvLayer {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {
    
    let result_height = (((input[0].rows() as f64 + 2.0* self.config.padding as f64 - self.filter_size.0 as f64)/self.config.stride as f64) + 1.0).floor() as usize;
    let result_width = (((input[0].cols() as f64 + 2.0* self.config.padding as f64 - self.filter_size.1 as f64)/self.config.stride as f64) + 1.0).floor() as usize;
    let mut result_final = Vec::new();
    let mut z1_final = Vec::new();

    println!("CNN Input Size (Forward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len());
    println!("CNN Input (Forward) = {:?}", input);

    for (f,b) in self.filters.iter().zip(self.bias.iter()) {
      let mut result_channels = Vec::new();
      let mut z1_channels = Vec::new();
      for (inp,fc) in input.iter().zip(f.iter()) {
        let mut result = Tensor::zeros(result_height, result_width);
        for i in (0..inp.rows() - self.filter_size.0).step_by(self.config.stride) {
          for j in (0..inp.cols() - self.filter_size.1).step_by(self.config.stride) {
            let mut sum = *b;
            for k in 0 .. self.filter_size.0 {
              for l in 0 .. self.filter_size.1 {
                sum += inp.get(i+k,j+l) * fc.get(k,l);
              }
            }
            result.set(i/self.config.stride, j/self.config.stride, sum);
          }
        }
        let z1 = self.config.activation.forward(&result);
        result_channels.push(z1.clone());
        z1_channels.push(Box::new(z1));
      }
      result_final.push(result_channels); 
      z1_final.push(z1_channels); 
    }

    let mut output = Vec::new();
    let mut z1 = Vec::new();
    for (i,z) in result_final.iter().zip(z1_final.iter()) {
      let final_result = i.iter()
                                .fold(Some(Tensor::zeros(result_height, result_width)), |a,b| Some(a.unwrap().add(b)))
                                .unwrap_or(Tensor::zeros(result_height, result_width));

      output.push(Box::new(final_result));

      let final_z1 = z.iter()
                                .fold(Some(Tensor::zeros(z[0].rows(), z[0].cols())), |a,b| Some(a.unwrap().add(b)))
                                .unwrap_or(Tensor::zeros(z[0].rows(), z[0].cols()));

      z1.push(Box::new(final_z1))
    }

    self.last_input = Some(input.clone());
    self.last_z1 = Some(z1.clone());

    //println!("CNN Filter (Forward) = {:?}", output);

    println!("CNN Filter Size (Forward) = {}x{}x{}", self.filters[0][0].rows(), self.filters[0][0].cols(), self.filters[0].len());
    println!("CNN Filter (Forward) = {:?}", self.filters);
    println!("CNN Output size (Forward) = {}x{}x{}", output[0].rows(), output[0].cols(), output.len());
    println!("CNN Output (Forward) = {:?}", output);


    Some(output)
  }

  fn backward(&mut self, input: &Vec<Box<Tensor>>, _: bool) -> Option<Vec<Box<Tensor>>> {

    let mut final_output = Vec::new();
    let mut final_dw = Vec::new();
    let mut final_db= Vec::new();
    
    println!("CNN Input (Backward) = {:?}", input);
    println!("CNN Input size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len());

    if let Some(ref lz1) = self.last_z1 {
      if let Some(ref forward_input) = self.last_input {
        for (((f,inp), b),z1) in self.filters.iter_mut().zip(input.iter()).zip(self.bias.iter()).zip(lz1.iter()) {
          let mut dw_channel = Vec::new();
          let row_pad = (forward_input[0].rows() - inp.rows())/2;
          let col_pad = (forward_input[0].cols() - inp.cols())/2;
          let mut output = inp.pad(row_pad, col_pad);
          let mut db = 0.0;
          for (fi,fc) in forward_input.iter().zip(f.iter_mut()) {

            let dz = inp.mul_wise(&self.config.activation.backward(&z1));
            let mut dw = Tensor::zeros(fc.rows(), fc.cols());

            for i in (0..fi.rows()-self.filter_size.0).step_by(self.config.stride) {
              for j in (0 .. fi.cols()-self.filter_size.1).step_by(self.config.stride) {
                for k in 0 .. self.filter_size.0 {
                  for l in 0 .. self.filter_size.1 {
                    output.set(i/self.config.stride,j/self.config.stride,output.get(i/self.config.stride,j/self.config.stride) + (inp.get(i/self.config.stride,j/self.config.stride) * fc.get(k,l)));
                    dw.set(k,l,dw.get(k,l) + fi.get(i+k, j+l) * inp.get(i/self.config.stride,j/self.config.stride));
                  }
                }
                db += inp.get(i/self.config.stride,j/self.config.stride);
              }
            }
            dw_channel.push(dw);
          }
          final_output.push(Box::new(self.config.activation.backward(&output.add_value(*b))));
          final_db.push(if db > f64::MAX {f64::MAX} else {db});
          final_dw.push(dw_channel);
        }
      }
    }

    println!("CNN final_dw (Backward) = {:?}", final_dw);
    println!("CNN final_db (Backward) = {:?}", final_db);

    for (((f,dw),b),db) in self.filters.iter_mut().zip(final_dw.iter()).zip(self.bias.iter_mut()).zip(final_db.iter()) {
      for (fc,dw_channel) in f.iter_mut().zip(dw.iter()) {
        for k in 0.. fc.rows() {
          for l in 0.. fc.cols() {
            fc.set(k,l,fc.get(k,l) - (dw_channel.get(k,l) * self.config.learn_rate));
            *b = *b - (db * self.config.learn_rate);
          }
        }
      }
    }
    println!("CNN Filters (Backward) = {:?}", self.filters);

    println!("CNN Output (Backward) = {:?}", final_output);
    println!("CNN Output size (Backward) = {}x{}x{}", final_output[0].rows(), final_output[0].cols(), final_output.len());

    Some(final_output)
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

impl LayerPropagation for FlattenLayer {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {

    self.input_rows = input[0].rows();
    self.input_cols = input[0].cols();
    self.n_channels = input.len();

    println!("FlattenLayer Input Size (Forward) = {}x{}x{}",input[0].rows(), input[0].cols(), input.len());
    println!("FlattenLayer Input (Forward) = {:?}",input);

    let mut tmp = Vec::new();
    for i in input.iter() {
      for j in 0..i.rows() {
        for k in 0..i.cols() {
          tmp.push(i.get(j,k));
        }
      }
    }

    let t = Tensor::from_data(tmp.len(), 1, tmp);
    println!("FlattenLayer Output Size (Forward) = {}x{}",t.rows(), t.cols());
    println!("FlattenLayer Output (Forward) = {:?}",t);
    Some(vec![Box::new(t)])
  }

  fn backward(&mut self, input: &Vec<Box<Tensor>>, _: bool) -> Option<Vec<Box<Tensor>>> {

    let mut output = Vec::new();

    println!("FlattenLayer Input (Backward) = {}x{}x{}",input[0].rows(), input[0].cols(), input.len());

    for n_channel in 0..self.n_channels {
      let mut tmp = Tensor::zeros(self.input_rows, self.input_cols);
      for i in 0..self.input_rows {
        for j in 0..self.input_cols {
          tmp.set(i,j, input[0].data()[(n_channel * (self.input_cols*self.input_rows)) + (i * self.input_cols + j)])
        }
      }
      output.push(Box::new(tmp));
    }

    println!("FlattenLayer Output Size (Backward) = {}x{}x{}",output[0].rows(), output[0].cols(),output.len());
    println!("FlattenLayer Output (Backward) = {:?}",output);

    Some(output)
  }
}

pub struct PoolingLayerConfig {
  pub stride: usize
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

impl LayerPropagation for PoolingLayer {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {
    
    let result_height = (((input[0].rows() as f64 - self.filter_size.0 as f64)/self.config.stride as f64) + 1.0).floor() as usize;
    let result_width = (((input[0].cols() as f64 - self.filter_size.1 as f64)/self.config.stride as f64) + 1.0).floor() as usize;
    
    println!("PoolingLayer Input (Forward) = {:?}", input);
    println!("PoolingLayer Input size (Forward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len());
    println!("PoolingLayer Output size (Forward) = {}x{}x{}", result_height, result_width, input.len());

    let mut result_final = Vec::new();
    for inp in input.iter() {
      let mut result = Tensor::zeros(result_height, result_width);
      for i in (0 .. inp.rows() - self.filter_size.0).step_by(self.config.stride) {
        for j in (0 .. inp.cols() - self.filter_size.1).step_by(self.config.stride) {
          let mut max = 0.0;
          for k in 0 .. self.filter_size.0 {
            for l in 0 .. self.filter_size.1 {
              let value = inp.get(i+k,j+l);
              if  value > max {
                max = value;
              }
            }
          }
          result.set(i/self.config.stride, j/self.config.stride, max);
        }
      }
      result_final.push(Box::new(result));
    }

    self.last_input = Some(input.clone());

    println!("PoolingLayer Output (Forward) = {:?}", result_final);

    Some(result_final)
  }

  fn backward(&mut self, input: &Vec<Box<Tensor>>, _: bool) -> Option<Vec<Box<Tensor>>> {

    let mut result_final = Vec::new();

    println!("Pooling Input (Backward) = {:?}", input);
    println!("Pooling Input size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len());

    if let Some(ref fic) = self.last_input {
      for (inp,fi) in input.iter().zip(fic.iter()) {
        let mut result =  Tensor::zeros(fi.rows(), fi.cols());
        for i in (0 .. fi.rows()-self.filter_size.0).step_by(self.config.stride) {
          for j in (0 .. fi.cols()-self.filter_size.1).step_by(self.config.stride) {
            let mut max = 0.0;
            let mut max_k = 0;
            let mut max_l = 0;
            for k in 0 .. self.filter_size.0 {
              for l in 0 .. self.filter_size.1 {
                let value = fi.get(i+k,j+l);
                if value > max {
                  max = value;
                  max_k = k;
                  max_l = l;
                }
              }
            }
            result.set(i+max_k,j+max_l, inp.get(i/self.config.stride,j/self.config.stride));
          }
        }
        result_final.push(Box::new(result));  
      }
    }

    println!("Pooling Output (Backward) = {:?}", result_final);
    println!("Pooling Output size (Backward) = {}x{}x{}", result_final[0].rows(), result_final[0].cols(), result_final.len());

    Some(result_final)
  }
}