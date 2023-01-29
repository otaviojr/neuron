use std::{time::Instant, result, ptr};
use opencl3::{program::Program, types::{cl_float, cl_int, cl_event, CL_NON_BLOCKING}, kernel::{Kernel, ExecuteKernel}, memory::{CL_MEM_READ_ONLY, Buffer}};
use crate::{math::{Tensor, opencl::OCL, MatrixMathExecutorEnum}, Neuron};
use super::{ConvLayerExecutor, cpu::{ConvLayerCPU, PoolingLayerCPU}, ConvLayerConfig, PoolingLayerExecutor, PoolingLayerConfig};

const CONV_PROGRAM_SOURCE: &str = r#"
__kernel void conv(__global float *input, __global float *filter, __global float *result, float bias, int input_width, int input_height, int filter_width, int filter_height, int result_width, int result_height, int stride, int padding) {
  int gid = get_global_id(0);

  int gid_y = gid / result_width;
  int gid_x = gid % result_width;

  int i = gid_y * stride;
  int j = gid_x * stride;

  float sum = bias;
  for(int k = -padding; k < filter_height + padding; k++) {
    for(int l = -padding; l < filter_width + padding; l++) {
      int filter_index = (k + padding) * (filter_width + 2 * padding) + (l + padding);
      int input_index = (i + k) * input_width + (j + l);
      if (i + k >= 0 && j + l >= 0 && i + k < input_height && j + l < input_width) {
        sum += input[input_index] * filter[filter_index];
      }
    }
  }
  result[gid] = sum;
}

__kernel void conv_full(__global float *input, __global float *filter, __global float *bias, __global float *result, int n_filters, int n_channels, int input_width, int input_height, int filter_width, int filter_height, int result_width, int result_height, int stride, int padding) {
  int gid = get_global_id(0);

  int filter_block_size = filter_width * filter_height * n_channels;

  int n_filter = gid / filter_block_size;
  int filter_pos = gid % filter_block_size;

  int channel_size = filter_width*filter_height;

  int n_channel = filter_pos / channel_size;
  int channel_pos = filter_pos % channel_size;

  int gid_y = channel_pos / result_width;
  int gid_x = channel_pos % result_width;

  int i = gid_y * stride;
  int j = gid_x * stride;

  float sum = bias[n_filter];
  for(int k = -padding; k < filter_height + padding; k++) {
    for(int l = -padding; l < filter_width + padding; l++) {
      int filter_index = ((k + padding) * (filter_width + 2 * padding) + (l + padding)) + (n_filter * filter_block_size) + (n_channel * channel_size);
      int input_index = ((i + k) * input_width + (j + l)) + (n_channel * input_width * input_height);
      if (i + k >= 0 && j + l >= 0 && i + k < input_height && j + l < input_width) {
        sum += input[input_index] * filter[filter_index];
      }
    }
  }
  result[gid] = sum;
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

  float max = -DBL_MAX;
  for(int k = 0; k < filter_height; k++) {
    for(int l = 0; l < filter_width; l++) {
      int input_index = (i + k) * input_width + (j + l) + block * input_width * input_height;
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
const KERNEL_FULL_CONV_NAME: &str = "conv_full";
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
      if let Ok(p) = Program::create_and_build_from_source(&matrix_ocl.get_ocl_context().unwrap(), CONV_PROGRAM_SOURCE, "") {
        program = Some(p);
      }
    }
    
    ConvLayerOCL {
      program,
      cpu: ConvLayerCPU::init()
    }
  }
}

impl ConvLayerOCL{
  fn do_full_conv(&self, input: &Vec<Box<Tensor>>, filters: Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: Vec<f32>, config: &ConvLayerConfig) -> Option<(Vec<Box<Tensor>>, Vec<Box<Tensor>>)> {
    let result_size = ((((input[0].rows() as f32 + 2.0* config.padding as f32 - filter_size.0 as f32)/config.stride as f32) + 1.0).floor() as usize,
                                      (((input[0].cols() as f32 + 2.0* config.padding as f32 - filter_size.1 as f32)/config.stride as f32) + 1.0).floor() as usize);

    let input_size = (input[0].rows() * input.len(), input[0].cols());
    
    let mut result = Tensor::new(result_size.0 * filters.len() * filters[0].len(), result_size.1);

    Neuron::logger().debug(|| format!("OpenCL conv full input"));


    let mut data = Vec::new();
    for i in input.iter() {
      data.extend(i.data());
    }
    let input_tensor = Tensor::from_data(input_size.0, input_size.1, data);

    Neuron::logger().debug(|| format!("OpenCL conv full filter"));

    let data: Vec<&f32> = filters.iter().flatten().map(|f| f.data()).flatten().collect::<Vec<&f32>>();
    let mut n_data = Vec::new();
    for i in data {
      n_data.push(*i);
    }
    let filter_tensor = Tensor::from_data(filters.len() * filters[0].len() * filters[0][0].rows(), filter_size.1, n_data);

    Neuron::logger().debug(|| format!("OpenCL conv execute"));

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
            let error = write_event.wait();
            if let Err(error) = error {
              Neuron::logger().error(|| format!("OpenCL Error: {:?}", error));
              std::process::exit(0);
            }    
      
            kernel = Kernel::create(&program, KERNEL_FULL_CONV_NAME).unwrap();
  
            kernel_event = unsafe {
              ExecuteKernel::new(&kernel)
                  .set_arg(&*input_buffer)
                  .set_arg(&*filter_buffer)
                  .set_arg(&bias_buffer)
                  .set_arg(&*result_buffer)
                  .set_arg(&(filters.len() as cl_int))
                  .set_arg(&(filters[0].len() as cl_int))
                  .set_arg(&(input[0].cols() as cl_int))
                  .set_arg(&(input[0].rows() as cl_int))
                  .set_arg(&(filter_size.1 as cl_int))
                  .set_arg(&(filter_size.0 as cl_int))
                  .set_arg(&(result_size.1 as cl_int))
                  .set_arg(&(result_size.0 as cl_int))
                  .set_arg(&(config.stride as cl_int))
                  .set_arg(&(config.padding as cl_int))
                  .set_global_work_size(result.rows()*result.cols())
                  .enqueue_nd_range(&queue).unwrap()
            }; 
            events.push(kernel_event.get()); 
          };
          result.sync_ocl_cpu_wait(events);
        }

        Neuron::logger().debug(|| format!("OpenCL conv full activation"));

        let z1 = result.clone();
        let result = config.activation.forward(&result).unwrap();

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

        let mut output_results = Vec::new();
        let mut output_z1 = Vec::new();

        let ret = matrix_ocl.add_ocl_bulk(result_tensors.len(),result_tensors);
        let chucks = ret.data().chunks(result_size.0 * result_size.1);
        for chuck in chucks {
          let new_tensor = Tensor::from_data(result_size.0, result_size.1, chuck.to_vec());
          output_results.push(Box::new(new_tensor));
        }

        let ret = matrix_ocl.add_ocl_bulk(z1_tensors.len(), z1_tensors);
        let chucks = ret.data().chunks(result_size.0 * result_size.1);
        for chuck in chucks {
          let new_tensor = Tensor::from_data(result_size.0, result_size.1, chuck.to_vec());
          output_z1.push(Box::new(new_tensor));
        }

        return Some((output_z1, output_results));
      }
    }

    Neuron::logger().debug(|| format!("OpenCL convolution result = {:?}", result));
    None
  }

  fn do_conv(&self, input: &Box<Tensor>, filter: &Tensor, bias: &f32, result: &mut Tensor, config: &ConvLayerConfig) {

    let executor = Neuron::matrix();
    if let MatrixMathExecutorEnum::OCL(ref matrix_ocl) = **executor {
      if let Some(ref queue) = matrix_ocl.get_ocl_queue() {
        if let Some(ref program) = self.program {
          let mut events:Vec<cl_event> = Vec::default();
          let kernel;
          let kernel_event;
          {
            let input_ocl = input.get_ocl_buffer();
            let filter_ocl = filter.get_ocl_buffer();
            let result_ocl = result.get_ocl_buffer();
  
            let input_buffer = input_ocl.lock().unwrap();
            let filter_buffer = filter_ocl.lock().unwrap();
            let result_buffer = result_ocl.lock().unwrap();
  
            kernel = Kernel::create(&program, KERNEL_CONV_NAME).unwrap();
  
            kernel_event = unsafe {
              ExecuteKernel::new(&kernel)
                  .set_arg(&*input_buffer)
                  .set_arg(&*filter_buffer)
                  .set_arg(&*result_buffer)
                  .set_arg(&(*bias as cl_float))
                  .set_arg(&(input.cols() as cl_int))
                  .set_arg(&(input.rows() as cl_int))
                  .set_arg(&(filter.cols() as cl_int))
                  .set_arg(&(filter.rows() as cl_int))
                  .set_arg(&(result.cols() as cl_int))
                  .set_arg(&(result.rows() as cl_int))
                  .set_arg(&(config.stride as cl_int))
                  .set_arg(&(config.padding as cl_int))
                  .set_global_work_size(result.rows()*result.cols())
                  .enqueue_nd_range(&queue).unwrap()
            }; 
            events.push(kernel_event.get()); 
          };
          result.sync_ocl_cpu_wait(events);
        }
      }
    }
    Neuron::logger().debug(|| format!("OpenCL convolution result = {:?}", result));
  }
}

impl ConvLayerExecutor for ConvLayerOCL {
  fn forward(&self, input: &Vec<Box<Tensor>>, filters: Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: Vec<f32>, config: &ConvLayerConfig) -> Option<(Vec<Box<Tensor>>, Vec<Box<Tensor>>, Vec<Box<Tensor>>)> {

    let timer = Instant::now();

    let (z1, result) = self.do_full_conv(input, filters, filter_size, bias, config).unwrap();
    Some((input.clone(), z1, result))

    /*let result_height = (((input[0].rows() as f32 + 2.0* config.padding as f32 - filter_size.0 as f32)/config.stride as f32) + 1.0).floor() as usize;
    let result_width = (((input[0].cols() as f32 + 2.0* config.padding as f32 - filter_size.1 as f32)/config.stride as f32) + 1.0).floor() as usize;


    let mut result_final = Vec::new();
    let mut z1_final = Vec::new();

    Neuron::logger().debug(|| format!("CNN Input Size (Forward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));
    Neuron::logger().debug(|| format!("CNN Input (Forward) = {:?}", input));

    for (f,b) in filters.iter().zip(bias.iter()) {
      let mut result_channels = Vec::new();
      let mut z1_channels = Vec::new();
      
      for (inp,fc) in input.iter().zip(f.iter()) {
        let mut result = Tensor::new(result_height, result_width);

        self.do_conv(inp, fc, b, &mut result, config);

        Neuron::logger().profiling(|| format!("ConvLayer Forward Time (Before Activation) = {}ms", timer.elapsed().as_millis()));
        let z1 = config.activation.forward(&mut result).unwrap();
        Neuron::logger().profiling(|| format!("ConvLayer Forward Time (After Activation) = {}ms", timer.elapsed().as_millis()));

        result_channels.push(z1.clone());
        z1_channels.push(result);
      }
      result_final.push(result_channels); 
      z1_final.push(z1_channels); 
    }

    Neuron::logger().profiling(|| format!("ConvLayer Forward Time (Before Sum) = {}ms", timer.elapsed().as_millis()));

    let mut output = Vec::new();
    let mut z1 = Vec::new();

    let executor = Neuron::matrix();
    if let MatrixMathExecutorEnum::OCL(ref matrix_ocl) = **executor {
      let ret = matrix_ocl.add_ocl_bulk(result_final.len(),result_final.iter_mut().flatten().collect::<Vec<_>>());
      let chucks = ret.data().chunks(result_final[0][0].rows() * result_final[0][0].cols());
      for chuck in chucks {
        let new_tensor = Tensor::from_data(result_final[0][0].rows(), result_final[0][0].cols(), chuck.to_vec());
        output.push(Box::new(new_tensor));
      }

      let ret = matrix_ocl.add_ocl_bulk(z1_final.len(), z1_final.iter_mut().flatten().collect::<Vec<_>>());
      let chucks = ret.data().chunks(z1_final[0][0].rows() * z1_final[0][0].cols());
      for chuck in chucks {
        let new_tensor = Tensor::from_data(z1_final[0][0].rows(), z1_final[0][0].cols(), chuck.to_vec());
        z1.push(Box::new(new_tensor));
      }
    }

    let last_input = input.clone();
    let last_z1 = z1.clone();

    Neuron::logger().debug(|| format!("CNN Filter Size (Forward) = {}x{}x{}", filters[0][0].rows(), filters[0][0].cols(), filters[0].len()));
    Neuron::logger().debug(|| format!("CNN Filter (Forward) = {:?}", filters));
    Neuron::logger().debug(|| format!("CNN Output size (Forward) = {}x{}x{}", output[0].rows(), output[0].cols(), output.len()));
    Neuron::logger().debug(|| format!("CNN Output (Forward) = {:?}", output));

    Neuron::logger().profiling(|| format!("ConvLayer Forward Time = {}ms", timer.elapsed().as_millis()));

    Some((last_input, last_z1, output))*/
  }

  fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, last_z1: &Vec<Box<Tensor>>, filters: &mut Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: &mut Vec<f32>, activate: bool, config: &ConvLayerConfig) -> Option<Vec<Box<Tensor>>> {
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
    Neuron::logger().profiling(|| format!("PoolingLayer Forward Time = {}ms", timer.elapsed().as_millis()));
    Some((input.clone(), result))
  }

  fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, filter_size: (usize, usize), activate: bool, config: &PoolingLayerConfig) -> Option<Vec<Box<Tensor>>> {
    self.cpu.backward(input, forward_input, filter_size, activate, config)
  }
}