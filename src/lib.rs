pub mod math;
pub mod layers;
pub mod activations;
pub mod cost;
pub mod pipeline;

use std::{sync::Mutex, any::Any, io::{Write, Read}};

use math::{Tensor, MatrixMath, MatrixMathCPU};

pub trait Propagation: Any {
  fn forward(&mut self, input: &Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>>;
  fn backward(&mut self, input: &Vec<Box<Tensor>>, first: bool) -> Option<Vec<Box<Tensor>>>;
  fn as_loader(&self) -> Option<&dyn Loader>;
  fn as_mut_loader(&mut self) -> Option<&mut dyn Loader>;
}

#[derive(Clone)]
pub struct Weigths {
  name: String,
  weights: Vec<Box<Tensor>>,
  bias: Vec<Box<Tensor>>
}

pub trait Loader: Any {
  fn get_name(&self) -> String;
  fn get_weights(&self) -> Vec<Weigths>;
  fn set_weights(&mut self, weights: Vec<Weigths>);
}

pub struct Neuron {
  pipelines: Vec<Mutex<Box<dyn Propagation>>>
}

impl Neuron {
  pub fn new() -> Self {
    Neuron {
      pipelines: Vec::new()
    }
  }

  pub fn forward(&self, input: Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {
    let mut i = Some(input);
    for layer in self.pipelines.iter() {
      if let Some(ref i1) = i {
        i = layer.lock().unwrap().forward(i1);
      }
    }
    i
  }

  pub fn backward(&self, input: Vec<Box<Tensor>>) -> Option<Vec<Box<Tensor>>> {
    let mut i = Some(input);
    for (index,layer) in self.pipelines.iter().rev().enumerate() {
      if let Some(ref i1) = i {
        i = layer.lock().unwrap().backward(i1, index==0);
      }
    }
    i
  }

  pub fn matrix_math() -> Box<dyn MatrixMath> {
    Box::new(MatrixMathCPU { })
  }

  pub fn add_pipeline(&mut self, layer: Mutex<Box<dyn Propagation>>) -> &mut Self {
    self.pipelines.push(layer);
    self
  }

  pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
    let mut file = std::fs::File::create(path)?;
    for pipeline in self.pipelines.iter() {
      if let Ok(pipeline) = pipeline.lock() {
        if let Some(loader) = pipeline.as_loader() {
          let weights = loader.get_weights();
          for weight in weights.iter() {
            file.write(weight.name.as_bytes())?;
            file.write(&[0])?;
            file.write(&(weight.weights.len() as u64).to_le_bytes())?;
            for w in weight.weights.iter() {
              let byte_vec : Vec<u8> = w.data().iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
              file.write(&(w.rows() as u64).to_le_bytes())?;
              file.write(&(w.cols() as u64).to_le_bytes())?;
              file.write(&byte_vec)?;
            }
            file.write(&(weight.bias.len() as u64).to_le_bytes())?;
            for b in weight.bias.iter() {
              let byte_vec : Vec<u8> = b.data().iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
              file.write(&(b.rows() as u64).to_le_bytes())?;
              file.write(&(b.cols() as u64).to_le_bytes())?;
              file.write(&byte_vec)?;
            }
          }
        }
      }
    }
    Ok(())
  }

  pub fn load(&mut self, path: &str)  -> Result<(), std::io::Error> {
    let mut file = std::fs::File::open(path)?;
    let mut buffer = Vec::new();
    let mut final_weigths = Vec::new();

    file.read_to_end(&mut buffer)?;
    let mut index = 0;
    while index < buffer.len() {
      let mut name = Vec::new();
      while buffer[index] != 0 {
        name.push(buffer[index]);
        index += 1;
      }
      index += 1;
      let name = String::from_utf8(name).unwrap();
      let mut weights = Vec::new();
      let mut bias = Vec::new();
      let mut size = 0;
      for i in 0..8 {
        size += (buffer[index+i] as u64) << (i*8);
      }
      index += 8;
      let mut rows = 0;
      for i in 0..8 {
        rows += (buffer[index+i] as u64) << (i*8);
      }
      index += 8;
      let mut cols = 0;
      for i in 0..8 {
        cols += (buffer[index+i] as u64) << (i*8);
      }
      index += 8;
      let mut data = Vec::new();
      for _ in 0..size {
        let mut value = 0.0;
        for i in 0..8 {
          value += ((buffer[index+i] as u64) << (i*8)) as f64;
        }
        data.push(value);
        index += 8;
      }
      println!("Loading weights: {}x{}-{}", rows, cols, data.len());
      weights.push(Box::new(Tensor::from_data(rows as usize, cols as usize, data)));

      size = 0;
      for i in 0..8 {
        size += (buffer[index+i] as u64) << (i*8);
      }
      index += 8;

      let mut rows = 0;
      for i in 0..8 {
        rows += (buffer[index+i] as u64) << (i*8);
      }
      index += 8;

      let mut cols = 0;
      for i in 0..8 {
        cols += (buffer[index+i] as u64) << (i*8);
      }
      index += 8;

      let mut data = Vec::new();
      for _ in 0..size {
        let mut value = 0.0;
        for i in 0..8 {
          value += ((buffer[index+i] as u64) << (i*8)) as f64;
        }
        data.push(value);
        index += 8;
      }
      println!("Loading bias: {}x{}-{}", rows, cols, data.len());
      bias.push(Box::new(Tensor::from_data(rows as usize, cols as usize, data)));

      final_weigths.push(Weigths {
        name: name,
        weights,
        bias
      });
    }

    for pipeline in self.pipelines.iter_mut() {
      if let Ok(mut pipeline) = pipeline.lock() {
        if let Some(loader) = pipeline.as_mut_loader() {
          loader.set_weights(final_weigths.clone());
        }
      }
    }

    Ok(())
  }
}
