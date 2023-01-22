use std::ptr;

use opencl3::{device::{Device, get_all_devices, CL_DEVICE_TYPE_GPU}, context::Context, command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, program::Program, memory::{CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, Buffer}, types::{cl_double, CL_BLOCKING, CL_NON_BLOCKING, cl_event}, kernel::{Kernel, ExecuteKernel}};

use crate::math::Tensor;

use super::{ConvLayerExecutor, cpu::ConvLayerCPU, ConvLayerConfig};

const PROGRAM_SOURCE: &str = r#"
__kernel void conv(__global double *input, __global double *filter, __global double *result, double bias, int input_width, int input_height, int filter_width, int filter_height, int result_width, int result_height, int stride) {
  int gid = get_global_id(0);

  int gid_x = gid % width;
  int gid_y = gid / width;

  int input_x = gid_x * stride;
  int input_y = gid_y * stride;

  double sum = bias;
  for(int i_y = 0; i_y < filter_height; i_y++) {
    for(int i_x = 0; i_x < filter_width; i_x++) {
      int filter_index = i_y * filter_width + i_x;
      int input_index = (input_y + i_y) * input_width + (input_x + i_x);
      sum += input[input_index] * filter[fulter_index];
    }
  }
  int result_index = gid_x + gid_y * result_width;
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
      println!("OpenCL device: {}", d.name().unwrap());
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
          let mut input_buffer = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_READ_ONLY, input.data().len(), ptr::null_mut()).unwrap()
          };
          let mut filter_buffer = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_READ_ONLY, filter.data().len(), ptr::null_mut()).unwrap()
          };
          let result_buffer = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_WRITE_ONLY, result.data().len(), ptr::null_mut()).unwrap()
          };  

          let _ = unsafe { queue.enqueue_write_buffer(&mut input_buffer, CL_BLOCKING, 0, input.data(), &[]).unwrap() };
          let write_event = unsafe { queue.enqueue_write_buffer(&mut filter_buffer, CL_NON_BLOCKING, 0, filter.data(), &[]).unwrap() };

          let kernel = Kernel::create(&program, KERNEL_MATRIX_CONV_NAME).unwrap();

          let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&input_buffer)
                .set_arg(&filter_buffer)
                .set_arg(&result_buffer)
                .set_arg(&bias)
                .set_arg(&(input.cols() as i32))
                .set_arg(&(input.rows() as i32))
                .set_arg(&(filter.cols() as i32))
                .set_arg(&(filter.rows() as i32))
                .set_arg(&(result.cols() as i32))
                .set_arg(&(result.rows() as i32))
                .set_arg(&(config.stride as i32))
                .set_global_work_size(input.rows()-config.stride * input.cols()-config.stride)
                .set_wait_event(&write_event)
                .enqueue_nd_range(&queue).unwrap()
          };

          let mut events: Vec<cl_event> = Vec::default();
          events.push(kernel_event.get());
          
          let ret = unsafe { queue.enqueue_read_buffer(&result_buffer, CL_NON_BLOCKING, 0, &mut result.mut_data(), &events).unwrap() };
          let error = ret.wait();

          if let Err(error) = error {
            println!("OpenCL Error: {:?}", error);
            std::process::exit(0);
          }
        }
      }
    }
    println!("OpenCL convalution = {:?}", result);
  }
}

impl ConvLayerExecutor for ConvLayerOCL {
  fn forward(&self, input: &Vec<Box<Tensor>>, filters: Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: Vec<f64>, config: &ConvLayerConfig) -> Option<(Vec<Box<Tensor>>, Vec<Box<Tensor>>, Vec<Box<Tensor>>)> {
    let result_height = (((input[0].rows() as f64 + 2.0* config.padding as f64 - filter_size.0 as f64)/config.stride as f64) + 1.0).floor() as usize;
    let result_width = (((input[0].cols() as f64 + 2.0* config.padding as f64 - filter_size.1 as f64)/config.stride as f64) + 1.0).floor() as usize;
    let mut result_final = Vec::new();
    let mut z1_final = Vec::new();

    println!("CNN Input Size (Forward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len());
    println!("CNN Input (Forward) = {:?}", input);

    for (f,b) in filters.iter().zip(bias.iter()) {
      let mut result_channels = Vec::new();
      let mut z1_channels = Vec::new();
      for (inp,fc) in input.iter().zip(f.iter()) {
        let mut result = Tensor::zeros(result_height, result_width);
        self.do_conv(inp, fc, b, &mut result, config);
        let z1 = config.activation.forward(&result);
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
                                .fold(Some(Tensor::zeros(result_height, result_width)), |a,b| Some(a.unwrap().add(b)))
                                .unwrap_or(Tensor::zeros(result_height, result_width));

      output.push(Box::new(final_result));

      let final_z1 = z.iter()
                                .fold(Some(Tensor::zeros(z[0].rows(), z[0].cols())), |a,b| Some(a.unwrap().add(b)))
                                .unwrap_or(Tensor::zeros(z[0].rows(), z[0].cols()));

      z1.push(Box::new(final_z1))
    }

    let last_input = input.clone();
    let last_z1 = z1.clone();

    //println!("CNN Filter (Forward) = {:?}", output);

    println!("CNN Filter Size (Forward) = {}x{}x{}", filters[0][0].rows(), filters[0][0].cols(), filters[0].len());
    println!("CNN Filter (Forward) = {:?}", filters);
    println!("CNN Output size (Forward) = {}x{}x{}", output[0].rows(), output[0].cols(), output.len());
    println!("CNN Output (Forward) = {:?}", output);


    Some((last_input, last_z1, output))
  }

  fn backward(&self, input: &Vec<Box<Tensor>>, forward_input: &Vec<Box<Tensor>>, last_z1: &Vec<Box<Tensor>>, filters: &mut Vec<Vec<Tensor>>, filter_size: (usize, usize), bias: &mut Vec<f64>, _: bool, config: &ConvLayerConfig) -> Option<Vec<Box<Tensor>>> {
    let mut final_output = Vec::new();
    let mut final_dw = Vec::new();
    let mut final_db= Vec::new();
    
    println!("CNN Input (Backward) = {:?}", input);
    println!("CNN Input size (Backward) = {}x{}x{}", input[0].rows(), input[0].cols(), input.len());

    for (((f,inp), b),z1) in filters.iter_mut().zip(input.iter()).zip(bias.iter()).zip(last_z1.iter()) {
      let mut dw_channel = Vec::new();
      let row_pad = (forward_input[0].rows() - inp.rows())/2;
      let col_pad = (forward_input[0].cols() - inp.cols())/2;
      let mut output = inp.pad(row_pad, col_pad);
      let mut db = 0.0;
      for (fi,fc) in forward_input.iter().zip(f.iter_mut()) {

        let dz = inp.mul_wise(&config.activation.backward(&z1));
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
      final_output.push(Box::new(config.activation.backward(&output.add_value(*b))));
      final_db.push(db);
      final_dw.push(dw_channel);
    }

    println!("CNN final_dw (Backward) = {:?}", final_dw);
    println!("CNN final_db (Backward) = {:?}", final_db);

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
    println!("CNN Filters (Backward) = {:?}", filters);

    println!("CNN Output (Backward) = {:?}", final_output);
    println!("CNN Output size (Backward) = {}x{}x{}", final_output[0].rows(), final_output[0].cols(), final_output.len());

    Some(final_output)  
  }
}