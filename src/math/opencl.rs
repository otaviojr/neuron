use opencl3::{memory::{Buffer, CL_MEM_READ_WRITE}, context::Context, kernel::{Kernel, ExecuteKernel}, device::{Device, get_all_devices, CL_DEVICE_TYPE_GPU}, command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, program::{Program, CL_STD_3_0}, types::{cl_float, CL_NON_BLOCKING, cl_event, cl_int}};
use std::{ptr, time::Instant, sync::{Arc, Mutex}};
use crate::Neuron;

use super::{MatrixMathExecutor, Tensor, MatrixMathExecutorEnum};

const PROGRAM_SOURCE: &str = r#"
__kernel void add(__global float *a, __global float *b, int width) {
  int gid = get_global_id(0);
  a[gid] = a[gid] + b[gid];
}

__kernel void add_bulk(__global float *a, __global float *b, int blocks, int len, int width, int height) {
  int gid = get_global_id(0);

  int block = gid / (width * height);
  int pos = gid % (width * height);

  int result_index = pos + (block * width * height); 
  
  float sum = 0.0;
  for(int i = 0; i < len; i++){
    int source_index = pos + (i * width * height) + (block * len * width * height);
    sum += a[source_index];
  }

  b[result_index] = sum;
}

__kernel void sub(__global float *a, __global float *b, __global float *c, int width) {
  int gid = get_global_id(0);
  c[gid] = a[gid] - b[gid];
}

__kernel void div(__global float *a, __global float *b, __global float *c, int width) {
  int gid = get_global_id(0);
  c[gid] = a[gid] / b[gid];
}

__kernel void mul_wise(__global float *a, __global float *b, int width) {
  int gid = get_global_id(0);
  a[gid] = a[gid] * b[gid];
}

__kernel void mul(__global float *a, __global float *b, __global float *c, int width_a, int width_b, int width_c) {
  int gid = get_global_id(0);

  int row = gid / width_c;
  int col = gid % width_c;

  float sum = 0.0;
  for (int i = 0; i < width_a; i++) {
      sum += a[row * width_a + i] * b[i * width_b + col];
  }
  c[gid] = sum;
}

__kernel void transpose(__global float *a, int width, int height) {
  int gid = get_global_id(0);
  int row = gid / width;
  int col = gid % width;

  if (row < col) {
    int temp = a[row * width + col];
    a[row * width + col] = a[col * width + row];
    a[col * width + row] = temp;
  }
}

__kernel void add_value(__global float *a, float value) {
  int gid = get_global_id(0);
  a[gid] = a[gid] + value;
}

__kernel void mul_value(__global float *a, float value) {
  int gid = get_global_id(0);
  a[gid] = a[gid] * value;
}

__kernel void div_value(__global float *a, float value) {
  int gid = get_global_id(0);
  a[gid] = a[gid] / value;
}

__kernel void zero(__global float *a) {
  int gid = get_global_id(0);
  a[gid] = 0;
}
"#;

const KERNEL_MATRIX_ADD_NAME: &str = "add";
const KERNEL_MATRIX_ADD_BULK_NAME: &str = "add_bulk";
const KERNEL_MATRIX_ADD_VALUE_NAME: &str = "add_value";
const KERNEL_MATRIX_SUB_NAME: &str = "sub";
const KERNEL_MATRIX_DIV_NAME: &str = "div";
const KERNEL_MATRIX_DIV_VALUE_NAME: &str = "div_value";
const KERNEL_MATRIX_MUL_NAME: &str = "mul";
const KERNEL_MATRIX_MUL_VALUE_NAME: &str = "mul_value";
const KERNEL_MATRIX_MUL_WISE_NAME: &str = "mul_wise";
const KERNEL_MATRIX_TRANSPOSE_NAME: &str = "transpose";
const KERNEL_MATRIX_ZERO_NAME: &str = "zero";

pub struct MatrixMathOCL {
  device: Option<Device>,
  context: Option<Context>,
  queue: Option<CommandQueue>,
  program: Option<Program>,
}

impl MatrixMathOCL {
  pub fn init() -> Self {
    let mut device = None;
    let mut context = None;
    let mut queue = None;
    let mut program = None;

    match get_all_devices(CL_DEVICE_TYPE_GPU) {
      Ok(device_id) => {
        let d = Device::new(device_id.first().unwrap().clone());

        Neuron::logger().info(|| format!("OpenCL device (MatrixMathOCL): {}", d.name().unwrap()));

        match Context::from_device(&d) {
          Ok(c) => {
            match CommandQueue::create_default_with_properties(&c, 0, 0) {
              Ok(q) => {
                match Program::create_and_build_from_source(&c, PROGRAM_SOURCE, "") {
                  Ok(p) => {
                    device= Some(d);
                    context = Some(c);
                    queue = Some(q);
                    program = Some(p);
                  },
                  Err(e) => {
                    println!("OpenCL Error: {:?}", e);
                    std::process::exit(0);                
                  }
                }
              },
              Err(e) => {
                println!("OpenCL Error: {:?}", e);
                std::process::exit(0);                
              }
            }
          },
          Err(e) => {
            println!("OpenCL Error: {:?}", e);
            std::process::exit(0);                
          }
        }
      },
      Err(e) => {
        println!("OpenCL Error: {:?}", e);
        std::process::exit(0);                
      }
    }

    MatrixMathOCL {
      device,
      context,
      queue,
      program,
    }
  }

  pub fn get_ocl_context(&self) -> Option<&Context> {
    self.context.as_ref()
  }

  pub fn get_ocl_queue(&self) -> Option<&CommandQueue> {
    self.queue.as_ref()
  }

  pub fn add_ocl_bulk(&self, blocks: usize, a: Vec<Box<Tensor>>) -> Tensor {
    //Create a new tensor to store the result

    let len = a.len() / blocks;
    let input_size = (a[0].rows * a.len(), a[0].cols);
    let output_size = (a[0].rows * blocks, a[0].cols);
    let single_output_size = (a[0].rows, a[0].cols);

    let mut result = Tensor::new(output_size.0, output_size.1);

    let mut data = Vec::new();
    for a in a {
      data.extend(a.data());
    }
    let input = Tensor::from_data(input_size.0, input_size.1, data);

    if let Some(ref queue) = self.queue {
      if let Some(ref program) = self.program {
        let mut events:Vec<cl_event> = Vec::default();
        let kernel;
        let kernel_event;
        {  
          let i_ocl = input.get_ocl_buffer();
          let ib = i_ocl.lock().unwrap();
          let r_ocl = result.get_ocl_buffer();
          let rb = r_ocl.lock().unwrap();
      
          kernel = Kernel::create(program, KERNEL_MATRIX_ADD_BULK_NAME).unwrap();
  
          kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ib)
                .set_arg(&*rb)
                .set_arg(&(blocks as cl_int))
                .set_arg(&(len as cl_int))
                .set_arg(&(single_output_size.1 as cl_int))
                .set_arg(&(single_output_size.0 as cl_int))
                .set_global_work_size(result.cols * result.rows)
                .enqueue_nd_range(queue).unwrap()
          };
          events.push(kernel_event.get());
        };
        result.sync_ocl_cpu_wait(events);
      }
    }
    Neuron::logger().debug(|| format!("OpenCL add bulk matrix = {:?}", result));
    result
  }
}

impl MatrixMathExecutor for MatrixMathOCL {  
  fn add(&self, a: &mut Tensor, b: &Tensor) -> Tensor {
      // Check that the tensors are the same size
      assert!(a.rows == b.rows && a.cols == b.cols);

      if let Some(ref queue) = self.queue {
        if let Some(ref program) = self.program {
          let mut events:Vec<cl_event> = Vec::default();
          let kernel;
          let kernel_event;  
          {
            let a_ocl = a.get_ocl_buffer();
            let b_ocl = b.get_ocl_buffer();
  
            let ab = a_ocl.lock().unwrap();
            let bb = b_ocl.lock().unwrap();
  
            kernel = Kernel::create(&program, KERNEL_MATRIX_ADD_NAME).unwrap();
  
            kernel_event = unsafe {
              ExecuteKernel::new(&kernel)
                  .set_arg(&*ab)
                  .set_arg(&*bb)
                  .set_arg(&(a.cols as cl_int))
                  .set_global_work_size(a.cols * a.rows)
                  .enqueue_nd_range(&queue).unwrap()
            };  
            events.push(kernel_event.get());
          };
          a.sync_ocl_cpu_wait(events);
        }
      }

      Neuron::logger().debug(|| format!("OpenCL add matrix = {:?}", a));
      a.clone()
  }

  fn sub(&self, a: &mut Tensor, b: &Tensor) -> Tensor {
    // Check that the tensors are the same size
    assert!(a.rows == b.rows && a.cols == b.cols);

    if let Some(ref queue) = self.queue {
      if let Some(ref program) = self.program {
        let mut events:Vec<cl_event> = Vec::default();
        let kernel;
        let kernel_event;
        {
          let a_ocl = a.get_ocl_buffer();
          let b_ocl = b.get_ocl_buffer();
  
          let ab = a_ocl.lock().unwrap();
          let bb = b_ocl.lock().unwrap();
  
          kernel = Kernel::create(&program, KERNEL_MATRIX_SUB_NAME).unwrap();
  
          kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&*bb)
                .set_arg(&(a.cols as cl_int))
                .set_global_work_size(a.cols * a.rows)
                .enqueue_nd_range(&queue).unwrap()
          };  
          events.push(kernel_event.get());
        };
        a.sync_ocl_cpu_wait(events);
      }
    }

    Neuron::logger().debug(|| format!("OpenCL sub matrix = {:?}", a));
    a.clone()
  }

  fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
    // Check that the tensors are compatible for multiplication
    assert!(a.cols == b.rows);

    let timer = Instant::now();

    // Create a new tensor to store the result
    let mut result = Tensor::new(a.rows, b.cols);

    Neuron::logger().profiling(|| format!("Matrix Mul (results created) = {}ns", timer.elapsed().as_millis()));

    if let Some(ref queue) = self.queue {
      if let Some(ref program) = self.program {
        let mut events:Vec<cl_event> = Vec::default();
        let kernel;
        let kernel_event;
        {
          let a_ocl = a.get_ocl_buffer();
          let b_ocl = b.get_ocl_buffer();
          let r_ocl = result.get_ocl_buffer();
  
          let ab = a_ocl.lock().unwrap();
          let bb = b_ocl.lock().unwrap();
          let rb = r_ocl.lock().unwrap();
  
          Neuron::logger().profiling(|| format!("Matrix Mul (opencl memory) = {}ns", timer.elapsed().as_millis()));
  
          kernel = Kernel::create(&program, KERNEL_MATRIX_MUL_NAME).unwrap();
  
          Neuron::logger().profiling(|| format!("Matrix Mul (opencl kernel created) = {}ns", timer.elapsed().as_millis()));
  
          kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&*bb)
                .set_arg(&*rb)
                .set_arg(&(a.cols as cl_int))
                .set_arg(&(b.cols as cl_int))
                .set_arg(&(result.cols as cl_int))
                .set_global_work_size(result.cols * result.rows)
                .enqueue_nd_range(&queue).unwrap()
          };  
          events.push(kernel_event.get());
        };
        result.sync_ocl_cpu_wait(events);

        Neuron::logger().profiling(|| format!("Matrix Mul (opencl result readed) = {}ns", timer.elapsed().as_millis()));
      }
    }

    Neuron::logger().debug(|| format!("OpenCL multiply matrix = {:?}", result));
    result
  }

  fn mul_wise(&self, a: &mut Tensor, b: &Tensor) -> Tensor {
    // Check that the tensors are compatible for multiplication
    assert!(a.rows == b.rows && a.cols == b.cols);

    if let Some(ref queue) = self.queue {
      if let Some(ref program) = self.program {
        let mut events:Vec<cl_event> = Vec::default();
        let kernel;
        let kernel_event;
        {
          let a_ocl = a.get_ocl_buffer();
          let b_ocl = b.get_ocl_buffer();
  
          let ab = a_ocl.lock().unwrap();
          let bb = b_ocl.lock().unwrap();
  
          kernel = Kernel::create(&program, KERNEL_MATRIX_MUL_WISE_NAME).unwrap();
  
          kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&*bb)
                .set_arg(&(a.cols as cl_int))
                .set_global_work_size(a.cols * a.rows)
                .enqueue_nd_range(&queue).unwrap()
          };  
          events.push(kernel_event.get());
        };
        a.sync_ocl_cpu_wait(events);
      }
    }

    Neuron::logger().debug(|| format!("OpenCL multiply wise matrix = {:?}", a));
    a.clone()
  }

  fn div(&self, a: &mut Tensor, b: &Tensor) -> Tensor {
      // Check that the tensors are the same size
      assert!(a.rows == b.rows && a.cols == b.cols);

      if let Some(ref queue) = self.queue {
        if let Some(ref program) = self.program {
          let mut events:Vec<cl_event> = Vec::default();
          let kernel;
          let kernel_event;  
          {
            let a_ocl = a.get_ocl_buffer();
            let b_ocl = b.get_ocl_buffer();
  
            let ab = a_ocl.lock().unwrap();
            let bb = b_ocl.lock().unwrap();
  
            kernel = Kernel::create(&program, KERNEL_MATRIX_DIV_NAME).unwrap();
  
            kernel_event = unsafe {
              ExecuteKernel::new(&kernel)
                  .set_arg(&*ab)
                  .set_arg(&*bb)
                  .set_arg(&(a.cols as cl_int))
                  .set_global_work_size(a.cols * a.rows)
                  .enqueue_nd_range(&queue).unwrap()
            };  
            events.push(kernel_event.get());
          };
          a.sync_ocl_cpu_wait(events);
        }
      }
      Neuron::logger().debug(|| format!("OpenCL div matrix = {:?}", a));
      a.clone()
  }

  fn dot(&self, a: &Tensor, b: &Tensor) -> f32 {
    let mut result = 0.0;

    // Check that the tensors are compatible for the dot product
    assert!(a.rows == b.rows && a.cols == b.cols);

    // Calculate the dot product
    for i in 0..a.rows {
        for j in 0..a.cols {
            result += a.get(i, j) * b.get(i, j);
        }
    }

    result
  }

  fn transpose(&self, a: &mut Tensor) -> Tensor {
    if let Some(ref queue) = self.queue {
      if let Some(ref program) = self.program {
        let mut events:Vec<cl_event> = Vec::default();
        let kernel;
        let kernel_event;
        {
          let a_ocl = a.get_ocl_buffer();
          let ab = a_ocl.lock().unwrap();
  
          kernel = Kernel::create(&program, KERNEL_MATRIX_TRANSPOSE_NAME).unwrap();
  
          kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&(a.cols as cl_int))
                .set_arg(&(a.rows as cl_int))
                .set_global_work_size(a.cols * a.rows)
                .enqueue_nd_range(&queue).unwrap()
          };  
          events.push(kernel_event.get());
        };
        a.sync_ocl_cpu_wait(events);
      }
    }

    Neuron::logger().debug(|| format!("OpenCL transpose matrix = {:?}", a));
    a.clone()
  }

  fn add_value(&self, a: &mut Tensor, value: f32) -> Tensor {
    if let Some(ref queue) = self.queue {
      if let Some(ref program) = self.program {
        let mut events:Vec<cl_event> = Vec::default();
        let kernel;
        let kernel_event;
        {
          let a_ocl = a.get_ocl_buffer();
          let ab = a_ocl.lock().unwrap();
  
          kernel = Kernel::create(&program, KERNEL_MATRIX_ADD_VALUE_NAME).unwrap();
  
          kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&(value as cl_float))
                .set_global_work_size(a.cols * a.rows)
                .enqueue_nd_range(&queue).unwrap()
          };  
          events.push(kernel_event.get());
        };
        a.sync_ocl_cpu_wait(events);
      }
    }
    Neuron::logger().debug(|| format!("OpenCL add value matrix = {:?}", a));
    a.clone()
  }

  fn div_value(&self, a: &mut Tensor, value: f32) -> Tensor {
    if let Some(ref queue) = self.queue {
      if let Some(ref program) = self.program {
        let mut events:Vec<cl_event> = Vec::default();
        let kernel;
        let kernel_event;
        {
          let a_ocl = a.get_ocl_buffer();  
          let ab = a_ocl.lock().unwrap();
  
          kernel = Kernel::create(&program, KERNEL_MATRIX_DIV_VALUE_NAME).unwrap();
  
          kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&(value as cl_float))
                .set_global_work_size(a.cols * a.rows)
                .enqueue_nd_range(&queue).unwrap()
          };  
          events.push(kernel_event.get());
        };
        a.sync_ocl_cpu_wait(events);
      }
    }
    Neuron::logger().debug(|| format!("OpenCL div value matrix = {:?}", a));
    a.clone()
  }

  fn mul_value(&self, a: &mut Tensor, value: f32) -> Tensor {
    if let Some(ref queue) = self.queue {
      if let Some(ref program) = self.program {
        let mut events: Vec<cl_event> = Vec::default();
        let kernel;
        let kernel_event;
        {
          let a_ocl = a.get_ocl_buffer();
          let ab = a_ocl.lock().unwrap();
  
          kernel = Kernel::create(&program, KERNEL_MATRIX_MUL_VALUE_NAME).unwrap();
  
          kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&(value as cl_float))
                .set_global_work_size(a.cols * a.rows)
                .enqueue_nd_range(&queue).unwrap()
          };  
          events.push(kernel_event.get());
        };
        a.sync_ocl_cpu_wait(events);
      }
    }

    Neuron::logger().debug(|| format!("OpenCL mul value matrix = {:?}", a));
    a.clone()
  }

  fn sum_row(&self, a:&Tensor) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::new(a.rows, 1);

    for i in 0..a.rows {
      let mut sum: f32 = 0.0;
      for j in 0..a.cols {
        sum += a.get(i, j);
      }
      result.set(i, 0, sum);
    }

    result.sync_cpu_ocl();
    result
  }

  fn broadcast(&self, a: &Tensor, rows: usize, cols: usize) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::new(rows, cols);

    if a.rows == rows {
      for i in 0..a.rows {
        for j in 0..result.cols {
          result.set(i, j, a.get(i, 0));
        }
      }  
    } else if a.cols == cols {
      for i in 0..result.rows {
        for j in 0..a.cols {
          result.set(i, j, a.get(0, j));
        }
      }  
    }
    result.sync_cpu_ocl();
    result
  }

  fn pad(&self, a: &Tensor, pad_row: usize, pad_col: usize) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::new(a.rows() + pad_row*2, a.cols() + pad_col*2);

    //println!("pad input = {}", a);

    for i in 0..a.rows() {
      for j in 0..a.cols() {
          result.set(i+pad_row, j+pad_col,a.get(i,j));
      }
    }

    //println!("pad output = {}", result);
    result.sync_cpu_ocl();
    result
  }

  fn zero(&self, a: &mut Tensor) -> Tensor {
    if let Some(ref queue) = self.queue {
      if let Some(ref program) = self.program {
        let mut events:Vec<cl_event> = Vec::default();
        let kernel;
        let kernel_event;
        {
          let a_ocl = a.get_ocl_buffer();
          let ab = a_ocl.lock().unwrap();
  
          kernel = Kernel::create(&program, KERNEL_MATRIX_ZERO_NAME).unwrap();
  
          kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_global_work_size(a.cols * a.rows)
                .enqueue_nd_range(&queue).unwrap()
          };  
          events.push(kernel_event.get());
        };
        a.sync_ocl_cpu_wait(events);
      }
    }

    Neuron::logger().debug(|| format!("OpenCL zero matrix = {:?}", a));
    a.clone()
  }
}

#[derive(Debug)]
pub struct TensorOCL {
  buffer: Arc<Mutex<Buffer<cl_float>>>,
}

impl TensorOCL {
  pub fn new(size: usize) -> Option<TensorOCL> {

    let timer = Instant::now();

    let executor = Neuron::matrix();
    if let MatrixMathExecutorEnum::OCL(ref matrix_ocl) = **executor {

      let ocl_buffer = unsafe {
        Buffer::<cl_float>::create(matrix_ocl.get_ocl_context().unwrap(), CL_MEM_READ_WRITE, size, ptr::null_mut()).unwrap()
      };
  
      Neuron::logger().profiling(|| format!("OpenCL Tensor (new) = {}ns", timer.elapsed().as_millis()));

      return Some(TensorOCL {
        buffer: Arc::new(Mutex::new(ocl_buffer)),
      });
    }
    None 
  }

  pub fn init(buffer: &Vec<f32>) -> Option<TensorOCL> {

    let timer = Instant::now();

    let executor = Neuron::matrix();
    if let MatrixMathExecutorEnum::OCL(ref matrix_ocl) = **executor {

      let mut ocl_buffer = unsafe {
        Buffer::<cl_float>::create(matrix_ocl.get_ocl_context().unwrap(), CL_MEM_READ_WRITE, buffer.len(), ptr::null_mut()).unwrap()
      };
  
      let write_event = unsafe { matrix_ocl.get_ocl_queue().unwrap().enqueue_write_buffer(&mut ocl_buffer, CL_NON_BLOCKING, 0, buffer, &[]).unwrap() };
      let error = write_event.wait();
  
      if let Err(error) = error {
        Neuron::logger().error(|| format!("OpenCL Error: {:?}", error));
        std::process::exit(0);
      }    
  
      //Neuron::logger().profiling(|| format!("OpenCL Tensor (init) = {}ns", timer.elapsed().as_millis()));

      return Some(TensorOCL {
        buffer: Arc::new(Mutex::new(ocl_buffer)),
      });
    }
    None 
  }
}

pub trait OCL {
  fn get_ocl_buffer(&self) -> Arc<Mutex<Buffer<cl_float>>>;
  fn sync_ocl_cpu(&mut self);
  fn sync_ocl_cpu_wait(&mut self, events: Vec<cl_event>);
  fn sync_cpu_ocl(&self);
}

impl OCL for Tensor {
  fn get_ocl_buffer(&self) -> Arc<Mutex<Buffer<cl_float>>> {
      let r = self.tensor_ocl.as_ref().unwrap().clone();
      let r1 = r.lock().unwrap();
      r1.buffer.clone()
  }

  fn sync_ocl_cpu(&mut self) {
    let executor = Neuron::matrix();
    if let MatrixMathExecutorEnum::OCL(ref matrix_ocl) = **executor {
      let buffer_ocl = self.get_ocl_buffer();
      let buffer = buffer_ocl.lock().unwrap();
        
      let ret = unsafe { matrix_ocl.get_ocl_queue().unwrap().enqueue_read_buffer(&buffer, CL_NON_BLOCKING, 0, &mut self.mut_data(), &[]).unwrap() };
      let error = ret.wait();
  
      if let Err(error) = error {
        println!("OpenCL Error: {:?}", error);
        std::process::exit(0);
      }
    }
  }

  fn sync_ocl_cpu_wait(&mut self, events: Vec<cl_event>) {
    let executor = Neuron::matrix();
    if let MatrixMathExecutorEnum::OCL(ref matrix_ocl) = **executor {
      let buffer_ocl = self.get_ocl_buffer();
      let buffer = buffer_ocl.lock().unwrap();
        
      let ret = unsafe { matrix_ocl.get_ocl_queue().unwrap().enqueue_read_buffer(&buffer, CL_NON_BLOCKING, 0, &mut self.mut_data(), &events).unwrap() };
      let error = ret.wait();
  
      if let Err(error) = error {
        println!("OpenCL Error: {:?}", error);
        std::process::exit(0);
      }
    }
  }

  fn sync_cpu_ocl(&self) {
    let executor = Neuron::matrix();
    if let MatrixMathExecutorEnum::OCL(ref matrix_ocl) = **executor {
      let buffer_ocl = self.get_ocl_buffer();
      let mut buffer = buffer_ocl.lock().unwrap();
        
      let ret = unsafe { matrix_ocl.get_ocl_queue().unwrap().enqueue_write_buffer(&mut buffer, CL_NON_BLOCKING, 0, &self.data(), &[]).unwrap() };
      let error = ret.wait();
  
      if let Err(error) = error {
        println!("OpenCL Error: {:?}", error);
        std::process::exit(0);
      }
    }
  }
}