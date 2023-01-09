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
  pub learn_rate: f64
}

pub struct LinearLayer {
  config: LinearLayerConfig,
  weights: Tensor,
  bias: Tensor,

  last_input: Option<Vec<Box<Tensor>>>,
  last_z1: Option<Tensor>
}

impl LinearLayer {
  pub fn new(input_size: usize, nodes: usize, config: LinearLayerConfig) -> Self {
    let mut rng = rand::thread_rng();

    LinearLayer {
      config: config,
      weights: Tensor::random(nodes,input_size),
      bias: Tensor::zeros(nodes,1),
      last_input: None,
      last_z1: None
    }
  }
}

impl LayerPropagation for LinearLayer {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {
    let input = &input[0];
    println!("Layer weights size = {}x{}", self.weights.rows(), self.weights.cols());
    println!("Bias = {}", self.bias);
    //println!("Input = {}", input);
    //println!("Weights = {}", self.weights);
    let z1_1 = self.weights.mul(&input);
    println!("z1_1 = {}x{}", z1_1.rows(), z1_1.cols());
    let b_bias = self.bias.broadcast(z1_1.rows(), z1_1.cols());
    println!("b_bias = {}x{}", b_bias.rows(), b_bias.cols());
    let z1 = z1_1.add(&b_bias);

    //println!("z1 = {}", z1);
    let ret = Some(vec![Box::new(self.config.activation.forward(&z1))]);
    self.last_z1 = Some(z1);
    self.last_input = Some(vec![input.clone()]);
    
    ret
  }

  fn backward(&mut self, input: &Vec<Box<Tensor>>, first: bool) -> Option<Vec<Box<Tensor>>> {
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
        let db = Tensor::from_data(dz.rows(), dz.cols(), dz.data().to_owned());
        println!("db size = {}x{}", db.rows(), db.cols());

        let ret = Some(vec![Box::new(self.weights.transpose().mul(&dz))]);

        self.weights = self.weights.sub(&dw.mul_value(self.config.learn_rate));
        let dbf = db.sum_row().div_value(forward_input.cols() as f64);
        self.bias = self.bias.sub(&dbf.mul_value(self.config.learn_rate));

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
  pub stride: usize
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
        let filter_channel = Tensor::random(filter_size.0, filter_size.1);
        filter_channels.push(filter_channel);
      }
      filters.push(filter_channels);
    }
  
    ConvLayer {
      config,
      filters,
      filter_size,
      bias: vec![0.0; n_filters],

      last_input: None,
    }
  }
}

impl LayerPropagation for ConvLayer {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {
    
    let result_height = (((input[0].rows() as f64 + 2.0* self.config.padding as f64 - self.filter_size.0 as f64)/self.config.stride as f64) + 1.0).floor() as usize;
    let result_width = (((input[0].cols() as f64 + 2.0* self.config.padding as f64 - self.filter_size.1 as f64)/self.config.stride as f64) + 1.0).floor() as usize;
    let mut result_final = Vec::new();

    for (f,b) in self.filters.iter().zip(self.bias.iter()) {
      let mut result_channels = Vec::new();
      for (i,fc) in input.iter().zip(f.iter()) {
        let mut result = Tensor::zeros(result_height, result_width);
        for y in (0..i.rows() - self.filter_size.0).step_by(self.config.stride) {
          for x in (0..i.cols() - self.filter_size.1).step_by(self.config.stride) {
            let mut sum = 0.0;
            for y1 in 0 .. self.filter_size.0 {
              for x1 in 0 .. self.filter_size.1 {
                sum += i.get(y+y1,x+x1) * fc.get(y1,x1);
              }
            }
            result.set(y, x, sum);
          }
        }
        let z1 = result.add_value(*b);
        result_channels.push(self.config.activation.forward(&z1));
      }
      result_final.push(result_channels); 
    }

    let mut output = Vec::new();
    for i in result_final.iter() {
      let final_result = i.iter()
                                .fold(Some(Tensor::zeros(result_height, result_width)), |a,b| Some(a.unwrap().add(b)))
                                .unwrap_or(Tensor::zeros(result_height, result_width));
      output.push(Box::new(final_result));
    }

    self.last_input = Some(input.clone());
    self.last_z1 = Some(output.clone());

    //println!("CNN Filter (Forward) = {:?}", output);

    println!("CNN Weights (Forward) = {}x{}x{}", self.filters[0][0].rows(), self.filters[0][0].cols(), self.filters[0].len());
    println!("CNN Output size (Forward) = {}x{}x{}", output[0].rows(), output[0].cols(), output.len());


    Some(output)
  }

  fn backward(&mut self, input: &Vec<Box<Tensor>>, _: bool) -> Option<Vec<Box<Tensor>>> {

    let mut final_output = Vec::new();
    let mut final_dw = Vec::new();
    let mut final_db= Vec::new();
    
    //println!("CNN Backward Input = {:?}", input);
    println!("CNN Input size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len());

    if let Some(z1) = self.last_z1 {
      for ((f,i), o) in self.filters.iter_mut().zip(input.iter()).zip(z1.iter()) {
        let mut dw_channel = Vec::new();
        let mut db = 0.0;
        if let Some(ref forward_input) = self.last_input {
          //println!("CNN Forward Input = {:?}", forward_input);
          for (fi,fc) in forward_input.iter().zip(f.iter_mut()) {
            let mut output = Tensor::zeros(fi.rows(), fi.cols());
            let mut dw = Tensor::zeros(fc.rows(), fc.cols());
            for y in (0..fi.rows()-self.filter_size.0).step_by(self.config.stride) {
              for x in (0 .. fi.cols()-self.filter_size.1).step_by(self.config.stride) {
                for y1 in 0 .. self.filter_size.0 {
                  for x1 in 0 .. self.filter_size.1 {
                    dw.set(y1,x1,self.config.activation.backward(&o).get(y,x) * fi.get(y+y1, x+x1));
                    output.set(y + y1, x + x1, self.config.activation.backward(&o).get(y,x) * fc.get(y1,x1));
                  }
                }
                db += i.get(y,x);
              }
            }
            final_output.push(Box::new(output));
            dw_channel.push(Box::new(dw));
          }
        }
        final_dw.push(dw_channel);
        final_db.push(db);
      }  
    }

    for (((f,dw),b),db) in self.filters.iter_mut().zip(final_dw.iter()).zip(self.bias.iter_mut()).zip(final_db.iter()) {
      for (fc,dw_channel) in f.iter_mut().zip(dw.iter()) {
        for y in 0.. fc.rows() {
          for x in 0.. fc.cols() {
            //println!("dw={}",dw_channel.get(y,x) * self.config.learn_rate);
            fc.set(y,x,fc.get(y,x) - (dw_channel.get(y,x) * self.config.learn_rate));
            *b = *b - (db * self.config.learn_rate); 
          }
        }
      }
    }

    //println!("CNN Filter (Backward) = {:?}", self.filters);
    //println!("CNN Filter (Backward) = {:?}", final_output);
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

    let mut tmp = Vec::new();
    for i in input.iter() {
      for j in 0..i.rows() {
        for k in 0..i.cols() {
          tmp.push(i.get(j,k));
        }
      }
    }

    let t = Tensor::from_data(tmp.len(), 1, tmp);
    println!("FlattenLayer Output = {}x{}",t.rows(), t.cols());
    Some(vec![Box::new(t)])
  }

  fn backward(&mut self, input: &Vec<Box<Tensor>>, _: bool) -> Option<Vec<Box<Tensor>>> {

    let mut output = Vec::new();

    println!("FlattenLayer Input = {}x{}x{}",input[0].rows(), input[0].cols(), input.len());

    for n_channel in 0..self.n_channels {
      let mut tmp = Tensor::zeros(self.input_rows, self.input_cols);
      for j in 0..self.input_rows {
        for k in 0..self.input_cols {
          tmp.set(j,k, input[0].data()[(n_channel + (self.input_cols*self.input_rows)) + (j * self.input_rows + k)])
        }
      }
      output.push(Box::new(tmp));
    }

    println!("FlattenLayer Output = {}x{}x{}",output[0].rows(), output[0].cols(),output.len());

    Some(output)
  }
}

pub struct PoolingLayerConfig {
  pub stride: usize
}

pub struct PoolingLayer {
  config: PoolingLayerConfig,
  n_channels: usize,
  filter_size: (usize,usize),

  last_input: Option<Vec<Box<Tensor>>>,
}

impl PoolingLayer {
  pub fn new(n_channels: usize, filter_size: (usize, usize), config: PoolingLayerConfig) -> Self {
    PoolingLayer {
      config,
      n_channels,
      filter_size,

      last_input: None
    }
  }
}

impl LayerPropagation for PoolingLayer {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {

    
    let result_height = (((input[0].rows() as f64 - self.filter_size.0 as f64)/self.config.stride as f64) + 1.0).floor() as usize;
    let result_width = (((input[0].cols() as f64 - self.filter_size.1 as f64)/self.config.stride as f64) + 1.0).floor() as usize;
    
    let mut result_final = Vec::new();

    for _ in 0..self.n_channels {
      let mut result_channels = Vec::new();
      for i in input.iter() {
        let mut result_y = 0;
        let mut result = Tensor::zeros(result_height, result_width);
        for y in (0 .. i.rows()-self.filter_size.0).step_by(self.config.stride) {
          let mut result_x = 0;
          for x in (0 .. i.cols()-self.filter_size.1).step_by(self.config.stride) {
            let mut max = 0.0;
            for y1 in 0 .. self.filter_size.0 {
              for x1 in 0 .. self.filter_size.1 {
                if i.get(y+y1,x+x1) > max {
                  max = i.get(y+y1,x+x1);
                }
              }
            }
            result.set(result_y, result_x, max);
            result_x += 1;
          }
          result_y += 1;
        }
        result_channels.push(result);
      }
      result_final.push(result_channels); 
    }

    let mut output = Vec::new();
    for i in result_final.iter() {
      let final_result = i.iter()
                                .fold(Some(Tensor::zeros(result_height, result_width)), |a,b| Some(a.unwrap().add(b)))
                                .unwrap_or(Tensor::zeros(result_height, result_width));
      output.push(Box::new(final_result));
    }

    self.last_input = Some(input.clone());

    println!("PoolingLayer Output (Forward) = {:?}", output);
    println!("Pooling Output size (Forward) = {}x{}x{}", output[0].rows(), output[0].cols(), output.len());

    Some(output)
  }

  fn backward(&mut self, input: &Vec<Box<Tensor>>, _: bool) -> Option<Vec<Box<Tensor>>> {

    let mut result_final = Vec::new();

    println!("Pooling Input size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len());

    if let Some(ref fic) = self.last_input {
      for i in input.iter() {
        for fi in fic.iter() {
          let mut result = Tensor::zeros(fi.rows(), fi.cols());
          for y in 0 .. i.rows() {
            for x in 0 .. i.cols() {
              let max = 0.0;
              for y1 in 0 .. self.filter_size.0 {
                for x1 in 0 .. self.filter_size.1 {
                  let value = fi.get(y+y1,x+x1);
                  if value > max {
                    result.set(y1,x1, result.get(y1,x1) + i.get(y, x));
                  }
                }
              }
            }
          }
          result_final.push(Box::new(result));  
        }
      }
    }

    println!("Pooling Output size (Backward) = {}x{}x{}", result_final[0].rows(), result_final[0].cols(), result_final.len());

    Some(result_final)
  }
}