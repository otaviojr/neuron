use std::{ptr, time::Instant};

use opencl3::{device::{Device, get_all_devices, CL_DEVICE_TYPE_GPU}, context::Context, command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, program::Program, memory::{CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, Buffer}, types::{cl_double, CL_BLOCKING, CL_NON_BLOCKING, cl_event}, kernel::{Kernel, ExecuteKernel}};

use crate::{math::{Tensor, opencl::OCL}, Neuron};

use super::{ConvLayerExecutor, cpu::ConvLayerCPU, ConvLayerConfig};

const PROGRAM_SOURCE: &str = r#"
__kernel void conv(__global double *input, __global double *filter, __global double *result, double bias, int input_width, int input_height, int filter_width, int filter_height, int result_width, int result_height, int stride, int padding) {
  int gid = get_global_id(0);

  int gid_y = gid / result_width;
  int gid_x = gid % result_width;

  int i = gid_y * stride;
  int j = gid_x * stride;

  double sum = bias;
  for(int k = -padding; k < filter_height + padding; k++) {
    for(int l = -padding; l < filter_width + padding; l++) {
      int filter_index = (k + padding) * (filter_width + 2 * padding) + (l + padding);
      int input_index = (i + k) * input_width + (j + l);
      if (i + k >= 0 && j + l >= 0 && i + k < input_height && j + l < input_width) {
        sum += input[input_index] * filter[filter_index];
      }
    }
  }
  int result_index = gid_y * result_width + gid_x;
  result[result_index] = sum;
}
"#;

const KERNEL_MATRIX_CONV_NAME: &str = "conv";

pub struct ConvLayerOCL {
  device: Option<Device>,
  context: Option<Context>,
  queue: Option<CommandQueue>,
  program: Option<Program>,
  cpu: ConvLayerCPU
}

impl ConvLayerOCL {
  pub fn init() -> Self {
    let mut device = None;
    let mut context = None;
    let mut queue = None;
    let mut program = None;

    if let Ok(device_id) = get_all_devices(CL_DEVICE_TYPE_GPU){
      let d = Device::new(device_id.first().unwrap().clone());
      Neuron::logger().info(|| format!("OpenCL device (ConvLayerOCL): {}", d.name().unwrap()));
      if let Ok(c) = Context::from_device(&d) {
        if let Ok(q) = CommandQueue::create_default(&c, CL_QUEUE_PROFILING_ENABLE) {
          if let Ok(p) = Program::create_and_build_from_source(&c, PROGRAM_SOURCE, "") {
            device= Some(d);
            context = Some(c);
            queue = Some(q);
            program = Some(p);
          }
        }
      }
    }

    ConvLayerOCL {
      device,
      context,
      queue,
      program,
      cpu: ConvLayerCPU::init()
    }
  }
}

impl ConvLayerOCL{
  fn do_conv(&self, input: &Box<Tensor>, filter: &Tensor, bias: &f64, result: &mut Tensor, config: &ConvLayerConfig) {
    if let Some(ref context) = self.context {
      if let Some(ref queue) = self.queue {
        if let Some(ref program) = self.program {
          let input_ocl = input.get_ocl_buffer();
          let filter_ocl = filter.get_ocl_buffer();

          let input_buffer = input_ocl.lock().unwrap();
          let filter_buffer = filter_ocl.lock().unwrap();
          let result_buffer = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_WRITE_ONLY, result.data().len(), ptr::null_mut()).unwrap()
          };  

          let kernel = Kernel::create(&program, KERNEL_MATRIX_CONV_NAME).unwrap();

          let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*input_buffer)
                .set_arg(&*filter_buffer)
                .set_arg(&result_buffer)
                .set_arg(&bias)
                .set_arg(&(input.cols() as i32))
                .set_arg(&(input.rows() as i32))
                .set_arg(&(filter.cols() as i32))
                .set_arg(&(filter.rows() as i32))
                .set_arg(&(result.cols() as i32))
                .set_arg(&(result.rows() as i32))
                .set_arg(&(config.stride as i32))
                .set_arg(&(config.padding as i32))
                .set_global_work_size(result.data().len())
                .enqueue_nd_range(&queue).unwrap()
          };

          let mut events: Vec<cl_event> = Vec::default();
          events.push(kernel_event.get());
          
          let ret = unsafe { queue.enqueue_read_buffer(&result_buffer, CL_NON_BLOCKING, 0, &mut result.mut_data(), &events).unwrap() };
          let error = ret.wait();

          if let Err(error) = error {
            Neuron::logger().error(|| format!("OpenCL Error: {:?}", error));
            std::process::exit(0);
          }
        }
      }
    }
    Neuron::logger().debug(|| format!("OpenCL convolution result = {:?}", result));
  }
}

impl ConvLayerExecutor for ConvLayerOCL {
  fn forward(&self, input: &Vec<Box<Tensor>>, filters: Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: Vec<f64>, config: &ConvLayerConfig) -> Option<(Vec<Box<Tensor>>, Vec<Box<Tensor>>, Vec<Box<Tensor>>)> {

    let timer = Instant::now();

    let result_height = (((input[0].rows() as f64 + 2.0* config.padding as f64 - filter_size.0 as f64)/config.stride as f64) + 1.0).floor() as usize;
    let result_width = (((input[0].cols() as f64 + 2.0* config.padding as f64 - filter_size.1 as f64)/config.stride as f64) + 1.0).floor() as usize;
    let mut result_final = Vec::new();
    let mut z1_final = Vec::new();

    Neuron::logger().debug(|| format!("CNN Input Size (Forward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len()));
    Neuron::logger().debug(|| format!("CNN Input (Forward) = {:?}", input));

    for (f,b) in filters.iter().zip(bias.iter()) {
      let mut result_channels = Vec::new();
      let mut z1_channels = Vec::new();
      for (inp,fc) in input.iter().zip(f.iter()) {
        let mut result = Tensor::zeros(result_height, result_width);

        self.do_conv(inp, fc, b, &mut result, config);
        
        let z1 = config.activation.forward(&mut result).unwrap();
        result_channels.push(z1.clone());
        z1_channels.push(Box::new(result));
      }
      result_final.push(result_channels); 
      z1_final.push(z1_channels); 
    }

    let mut output = Vec::new();
    let mut z1 = Vec::new();

    for (i,z) in result_final.iter_mut().zip(z1_final.iter_mut()) {
      let final_result = i.iter_mut()
                                .fold(Some(Tensor::zeros(result_height, result_width)), |a,b| Some(a.unwrap().add(b).unwrap()))
                                .unwrap_or(Tensor::zeros(result_height, result_width));

      output.push(Box::new(final_result));

      let z1_rows = z[0].rows();
      let z1_cols = z[0].cols();

      let final_z1 = z.iter_mut()
                                .fold(Some(Tensor::zeros(z1_rows, z1_cols)), |a,b| Some(a.unwrap().add(b).unwrap()))
                                .unwrap_or(Tensor::zeros(z1_rows, z1_cols));

      z1.push(Box::new(final_z1))
    }

    let last_input = input.clone();
    let last_z1 = z1.clone();

    Neuron::logger().debug(|| format!("CNN Filter Size (Forward) = {}x{}x{}", filters[0][0].rows(), filters[0][0].cols(), filters[0].len()));
    Neuron::logger().debug(|| format!("CNN Filter (Forward) = {:?}", filters));
    Neuron::logger().debug(|| format!("CNN Output size (Forward) = {}x{}x{}", output[0].rows(), output[0].cols(), output.len()));
    Neuron::logger().debug(|| format!("CNN Output (Forward) = {:?}", output));

    Neuron::logger().profiling(|| format!("ConvLayer Forward Time = {}ms", timer.elapsed().as_millis()));

    Some((last_input, last_z1, output))
  }

  fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, last_z1: &Vec<Box<Tensor>>, filters: &mut Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: &mut Vec<f64>, activate: bool, config: &ConvLayerConfig) -> Option<Vec<Box<Tensor>>> {
    self.cpu.backward(input, forward_input, last_z1, filters, filter_size, bias, activate, config)
  }
}