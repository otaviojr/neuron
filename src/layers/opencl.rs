use std::{time::Instant, result, ptr};
use opencl3::{program::{Program, CL_STD_3_0}, types::{cl_float, cl_int, cl_event, CL_NON_BLOCKING}, kernel::{Kernel, ExecuteKernel}, memory::{CL_MEM_READ_ONLY, Buffer, CL_MEM_READ_WRITE}};
use crate::{math::{Tensor, opencl::OCL, MatrixMathExecutorEnum}, Neuron};
use super::{ConvLayerExecutor, cpu::{ConvLayerCPU, PoolingLayerCPU}, ConvLayerConfig, PoolingLayerExecutor, PoolingLayerConfig};

const CONV_PROGRAM_SOURCE: &str = r#"
__kernel void conv(__global float *input, __global float *filter, __global float *bias, __global float *result, int n_channels, int input_width, int input_height, int filter_width, int filter_height, int result_width, int result_height, int stride, int padding) {
  int gid = get_global_id(0);

  int result_channel_size = result_width * result_height;
  int result_filter_block_size = result_channel_size * n_channels;

  int filter_channel_size = filter_width * filter_height;
  int filter_block_size = filter_channel_size * n_channels;

  int n_filter = gid / result_filter_block_size;
  int filter_pos = gid % result_filter_block_size;

  int n_channel = filter_pos / result_channel_size;
  int channel_pos = filter_pos % result_channel_size;

  int gid_y = channel_pos / result_width;
  int gid_x = channel_pos % result_width;

  int i = gid_y * stride;
  int j = gid_x * stride;

  float sum = bias[n_filter];
  for(int k = -padding; k < filter_height + padding; k++) {
    for(int l = -padding; l < filter_width + padding; l++) {
      if (i + k >= 0 && j + l >= 0 && i + k < input_height && j + l < input_width) {
        int filter_index = ((k + padding) * (filter_width + 2 * padding) + (l + padding)) + (n_filter * filter_block_size) + (n_channel * filter_channel_size);
        int input_index = ((i + k) * input_width + (j + l)) + (n_channel * input_width * input_height);
        sum += input[input_index] * filter[filter_index];
      }
    }
  }
  result[gid] = sum;
}

__kernel void conv_back(__global float *dz, __global float *forward_input, __global float *dw, __global float *db, __global float *filter, __global float *result, int n_channels, int input_width, int input_height, int filter_width, int filter_height, int result_width, int result_height, int stride, int padding) {
  int gid = get_global_id(0);

  int result_channel_size = result_width * result_height;
  int result_filter_block_size = result_channel_size * n_channels;

  int filter_channel_size = filter_width * filter_height;
  int filter_block_size = filter_channel_size * n_channels;

  int n_filter = gid / result_filter_block_size;
  int filter_pos = gid % result_filter_block_size;

  int n_channel = filter_pos / result_channel_size;
  int channel_pos = filter_pos % result_channel_size;

  int gid_y = channel_pos / result_width;
  int gid_x = channel_pos % result_width;

  int i = gid_y * stride;
  int j = gid_x * stride;

  for(int k = -padding; k < filter_height + padding; k++) {
    for(int l = -padding; l < filter_width + padding; l++) {
      int filter_index = ((k + padding) * (filter_width + 2 * padding) + (l + padding)) + (n_filter * filter_block_size) + (n_channel * filter_channel_size);
      int input_index = ((i + k) * input_width + (j + l)) + (n_channel * input_width * input_height);
      result[gid] += filter[filter_index] * dz[input_index];
      dw[filter_index] = forward_input[input_index] * dz[input_index];
    }
  }

  db[n_filter] += dz[gid];
}
"#;

const POOLING_PROGRAM_SOURCE: &str = r#"
__kernel void pooling(__global float *input, __global float *result, int input_width, int input_height, int filter_width, int filter_height, int result_width, int result_height, int stride) {
  int gid = get_global_id(0);

  int block_size = result_width * result_height;

  int block = gid / block_size;
  int pos = gid % block_size;

  int gid_y = pos / result_width;
  int gid_x = pos % result_width;

  int i = gid_y * stride;
  int j = gid_x * stride;

  float max = -FLT_MAX;
  for(int k = 0; k < filter_height; k++) {
    for(int l = 0; l < filter_width; l++) {
      int input_index = ((i + k) * input_width + (j + l)) + (block * input_width * input_height);
      float value = input[input_index];
      if(value > max) {
        max = value;
      }
    }
  }

  result[gid] = max;
}
"#;

const KERNEL_CONV_NAME: &str = "conv";
const KERNEL_POOLING_NAME: &str = "pooling";

pub struct ConvLayerOCL {
  program: Option<Program>,
  cpu: ConvLayerCPU
}

impl ConvLayerOCL {
  pub fn init() -> Self {
    let mut program = None;

    let executor = Neuron::matrix();
    if let MatrixMathExecutorEnum::OCL(ref matrix_ocl) = **executor {
      match Program::create_and_build_from_source(&matrix_ocl.get_ocl_context().unwrap(), CONV_PROGRAM_SOURCE, "") {
        Ok(p) => program = Some(p),
        Err(error) => {
          println!("OpenCL Error: {:?}", error);
          std::process::exit(0);
        }
      }
    }
    
    ConvLayerOCL {
      program,
      cpu: ConvLayerCPU::init()
    }
  }
}

impl ConvLayerOCL{
  fn do_conv(&self, input: &Vec<Box<Tensor>>, filters: Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: Vec<f32>, config: &ConvLayerConfig) -> Option<(Vec<Box<Tensor>>, Vec<Box<Tensor>>)> {
    
    let timer = Instant::now();
    
    let result_size = ((((input[0].rows() as f32 + 2.0* config.padding as f32 - filter_size.0 as f32)/config.stride as f32) + 1.0).floor() as usize,
                                      (((input[0].cols() as f32 + 2.0* config.padding as f32 - filter_size.1 as f32)/config.stride as f32) + 1.0).floor() as usize);

    let input_size = (input[0].rows() * input.len(), input[0].cols());
    
    let mut result = Tensor::new(result_size.0 * filters.len() * filters[0].len(), result_size.1);

    let mut data = Vec::new();
    for i in input.iter() {
      data.extend(i.data());
    }
    let input_tensor = Tensor::from_data(input_size.0, input_size.1, data);

    let data: Vec<&f32> = filters.iter().flatten().map(|f| f.data()).flatten().collect::<Vec<&f32>>();
    let mut n_data = Vec::new();
    for i in data {
      n_data.push(*i);
    }
    let filter_tensor = Tensor::from_data(filters.len() * filters[0].len() * filters[0][0].rows(), filter_size.1, n_data);

    let executor = Neuron::matrix();
    if let MatrixMathExecutorEnum::OCL(ref matrix_ocl) = **executor {
      if let Some(ref queue) = matrix_ocl.get_ocl_queue() {
        if let Some(ref program) = self.program {
          let mut events:Vec<cl_event> = Vec::default();
          let kernel;
          let kernel_event;
          {
            let input_ocl = input_tensor.get_ocl_buffer();
            let filter_ocl = filter_tensor.get_ocl_buffer();
            let result_ocl = result.get_ocl_buffer();
  
            let input_buffer = input_ocl.lock().unwrap();
            let filter_buffer = filter_ocl.lock().unwrap();
            let result_buffer = result_ocl.lock().unwrap();
  
            let mut bias_buffer = unsafe {
              Buffer::<cl_float>::create(matrix_ocl.get_ocl_context().unwrap(), CL_MEM_READ_ONLY, bias.len(), ptr::null_mut()).unwrap()
            };
        
            let write_event = unsafe { matrix_ocl.get_ocl_queue().unwrap().enqueue_write_buffer(&mut bias_buffer, CL_NON_BLOCKING, 0, &bias, &[]).unwrap() };
      
            kernel = Kernel::create(&program, KERNEL_CONV_NAME).unwrap();
  
            kernel_event = unsafe {
              ExecuteKernel::new(&kernel)
                  .set_arg(&*input_buffer)
                  .set_arg(&*filter_buffer)
                  .set_arg(&bias_buffer)
                  .set_arg(&*result_buffer)
                  .set_arg(&(input.len() as cl_int))
                  .set_arg(&(input[0].cols() as cl_int))
                  .set_arg(&(input[0].rows() as cl_int))
                  .set_arg(&(filter_size.1 as cl_int))
                  .set_arg(&(filter_size.0 as cl_int))
                  .set_arg(&(result_size.1 as cl_int))
                  .set_arg(&(result_size.0 as cl_int))
                  .set_arg(&(config.stride as cl_int))
                  .set_arg(&(config.padding as cl_int))
                  .set_global_work_size(result.rows()*result.cols())
                  .set_wait_event(&write_event)
                  .enqueue_nd_range(&queue).unwrap()
            }; 
            events.push(kernel_event.get());
          };
          result.sync_ocl_cpu_wait(events);
        }

        Neuron::logger().profiling(|| format!("ConvLayer Forward Time (Before Activation) = {}ns", timer.elapsed().as_nanos()));

        let z1 = result.clone();
        let result = config.activation.forward(&result).unwrap();

        Neuron::logger().profiling(|| format!("ConvLayer Forward Time (Before Normalization) = {}ns", timer.elapsed().as_nanos()));

        let mut result_tensors = Vec::new();
        let chucks = result.data().chunks(result_size.0 * result_size.1);
        for chuck in chucks {
          let new_tensor = Tensor::from_data(result_size.0, result_size.1, chuck.to_vec());
          result_tensors.push(Box::new(new_tensor));
        }

        let mut z1_tensors = Vec::new();
        let chucks = z1.data().chunks(result_size.0 * result_size.1);
        for chuck in chucks {
          let new_tensor = Tensor::from_data(result_size.0, result_size.1, chuck.to_vec());
          z1_tensors.push(Box::new(new_tensor));
        }

        Neuron::logger().profiling(|| format!("ConvLayer Forward Time (Before Sum) = {}ns", timer.elapsed().as_nanos()));

        let mut output_results = Vec::new();
        let mut output_z1 = Vec::new();

        let ret = matrix_ocl.add_ocl_bulk(filters.len(),result_tensors);
        let chucks = ret.data().chunks(result_size.0 * result_size.1);
        for chuck in chucks {
          let new_tensor = Tensor::from_data(result_size.0, result_size.1, chuck.to_vec());
          output_results.push(Box::new(new_tensor));
        }

        let ret = matrix_ocl.add_ocl_bulk(filters.len(), z1_tensors);
        let chucks = ret.data().chunks(result_size.0 * result_size.1);
        for chuck in chucks {
          let new_tensor = Tensor::from_data(result_size.0, result_size.1, chuck.to_vec());
          output_z1.push(Box::new(new_tensor));
        }

        Neuron::logger().debug(|| format!("OpenCL convolution result = {:?}", output_results));

        Neuron::logger().profiling(|| format!("ConvLayer Forward Time = {}ns", timer.elapsed().as_nanos()));

        return Some((output_z1, output_results));
      }
    }

    None
  }
}

impl ConvLayerExecutor for ConvLayerOCL {
  fn forward(&self, input: &Vec<Box<Tensor>>, filters: Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: Vec<f32>, config: &ConvLayerConfig) -> Option<(Vec<Box<Tensor>>, Vec<Box<Tensor>>, Vec<Box<Tensor>>)> {

    Neuron::logger().debug(|| format!("CNN Filter Size (Forward) = {}x{}x{}", filters[0][0].rows(), filters[0][0].cols(), filters[0].len()));
    Neuron::logger().debug(|| format!("CNN Filter (Forward) = {:?}", filters));

    let (z1, result) = self.do_conv(input, filters, filter_size, bias, config).unwrap();

    Neuron::logger().debug(|| format!("CNN Output size (Forward) = {}x{}x{}", result[0].rows(), result[0].cols(), result.len()));
    Neuron::logger().debug(|| format!("CNN Output (Forward) = {:?}", result));

    Some((input.clone(), z1, result))

  }

  fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, last_z1: &Vec<Box<Tensor>>, filters: &mut Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: &mut Vec<f32>, activate: bool, config: &ConvLayerConfig) -> Option<Vec<Box<Tensor>>> {

    //let result_size = (forward_input[0].rows() * filters.len() * filters[0].len(), forward_input[0].cols());
    //let mut result = Tensor::new(result_size.0, result_size.1);

    self.cpu.backward(input, forward_input, last_z1, filters, filter_size, bias, activate, config)
  }
}

pub struct PoolingLayerOCL {
  program: Option<Program>,
  cpu: PoolingLayerCPU
}

impl PoolingLayerOCL {
  pub fn init() -> Self {
    let mut program = None;

    let executor = Neuron::matrix();
    if let MatrixMathExecutorEnum::OCL(ref matrix_ocl) = **executor {
      if let Ok(p) = Program::create_and_build_from_source(&matrix_ocl.get_ocl_context().unwrap(), POOLING_PROGRAM_SOURCE, "") {
        program = Some(p);
      }
    }
    
    PoolingLayerOCL {
      program,
      cpu: PoolingLayerCPU::init()
    }
  }

  fn do_pooling(&self, input: &Vec<Box<Tensor>>, filter_size:(usize, usize), config: &PoolingLayerConfig) -> Result<Vec<Box<Tensor>>, String> {
    let result_size = ((((input[0].rows() as f32 - filter_size.0 as f32)/config.stride as f32) + 1.0).floor() as usize, 
                                       (((input[0].cols() as f32 - filter_size.1 as f32)/config.stride as f32) + 1.0).floor() as usize);

    Neuron::logger().debug(|| format!("PoolingLayer Output size (Forward) = {}x{}x{}", result_size.0, result_size.1, input.len()));

    let input_size = (input[0].rows() * input.len(), input[0].cols());

    let mut result = Tensor::new(result_size.0 * input.len(), result_size.1);

    let mut data = Vec::new();
    for i in input.iter() {
      data.extend(i.data());
    }
    let input_tensor = Tensor::from_data(input_size.0, input_size.1, data);

    let executor = Neuron::matrix();
    if let MatrixMathExecutorEnum::OCL(ref matrix_ocl) = **executor {
      if let Some(ref queue) = matrix_ocl.get_ocl_queue() {
        if let Some(ref program) = self.program {
          let mut events:Vec<cl_event> = Vec::default();
          let kernel;
          let kernel_event;
          {
            let input_ocl = input_tensor.get_ocl_buffer();
            let result_ocl = result.get_ocl_buffer();
  
            let input_buffer = input_ocl.lock().unwrap();
            let result_buffer = result_ocl.lock().unwrap();
  
            kernel = Kernel::create(&program, KERNEL_POOLING_NAME).unwrap();
  
            kernel_event = unsafe {
              ExecuteKernel::new(&kernel)
                  .set_arg(&*input_buffer)
                  .set_arg(&*result_buffer)
                  .set_arg(&(input[0].cols() as cl_int))
                  .set_arg(&(input[0].rows() as cl_int))
                  .set_arg(&(filter_size.1 as cl_int))
                  .set_arg(&(filter_size.0 as cl_int))
                  .set_arg(&(result_size.1 as cl_int))
                  .set_arg(&(result_size.0 as cl_int))
                  .set_arg(&(config.stride as cl_int))
                  .set_global_work_size(result.rows()*result.cols())
                  .enqueue_nd_range(&queue).unwrap()
            };
            events.push(kernel_event.get());
          };
          result.sync_ocl_cpu_wait(events);
        }
      }
    }

    let mut output = Vec::new();
    let chucks = result.data().chunks(result_size.0 * result_size.1);
    for chuck in chucks {
      let new_tensor = Tensor::from_data(result_size.0, result_size.1, chuck.to_vec());
      output.push(Box::new(new_tensor));
    }

    Neuron::logger().debug(|| format!("OpenCL pooling result = {:?}", result));
    Ok(output)
  }
}

impl PoolingLayerExecutor for PoolingLayerOCL {
  fn forward(&self, input: &Vec<Box<Tensor>>, filter_size: (usize, usize), config: &PoolingLayerConfig) -> Option<(Vec<Box<Tensor>>, Vec<Box<Tensor>>)>{
    let timer = Instant::now();

    Neuron::logger().debug(|| format!("PoolingLayer Input (Forward) = {:?}", input));
    Neuron::logger().debug(|| format!("PoolingLayer Input size (Forward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));

    let result = self.do_pooling(input, filter_size, config).unwrap();

    Neuron::logger().debug(|| format!("PoolingLayer Output (Forward) = {:?}", result));
    Neuron::logger().profiling(|| format!("PoolingLayer Forward Time = {}ns", timer.elapsed().as_millis()));
    Some((input.clone(), result))
  }

  fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, filter_size: (usize, usize), activate: bool, config: &PoolingLayerConfig) -> Option<Vec<Box<Tensor>>> {
    self.cpu.backward(input, forward_input, filter_size, activate, config)
  }
}