use opencl3::{memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY}, context::Context, kernel::{Kernel, ExecuteKernel}, device::{Device, get_all_devices, CL_DEVICE_TYPE_GPU}, command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, program::Program, types::{cl_double, CL_BLOCKING, CL_NON_BLOCKING, cl_ulong, cl_event, cl_int}};
use std::{ptr, time::Instant, sync::{Arc, Mutex}};
use crate::Neuron;

use super::{MatrixMathExecutor, Tensor, MatrixMathExecutorEnum};

const PROGRAM_SOURCE: &str = r#"
__kernel void add(__global double *a, __global double *b, __global double *c, int width) {
  int gid = get_global_id(0);
  int row = gid / width;
  int col = gid % width;
  c[gid] = a[gid] + b[gid];
}

__kernel void sub(__global double *a, __global double *b, __global double *c, int width) {
  int gid = get_global_id(0);
  int row = gid / width;
  int col = gid % width;
  c[gid] = a[gid] - b[gid];
}

__kernel void div(__global double *a, __global double *b, __global double *c, int width) {
  int gid = get_global_id(0);
  int row = gid / width;
  int col = gid % width;
  c[gid] = a[gid] / b[gid];
}

__kernel void mul_wise(__global double *a, __global double *b, __global double *c, int width) {
  int gid = get_global_id(0);
  int row = gid / width;
  int col = gid % width;
  c[gid] = a[gid] * b[gid];
}

__kernel void mul(__global double *a, __global double *b, __global double *c, int width_a, int width_b, int width_c) {
  int gid = get_global_id(0);
  int row = gid / width_c;
  int col = gid % width_c;
  double sum = 0.0;
  for (int i = 0; i < width_a; i++) {
      sum += a[row * width_a + i] * b[i * width_b + col];
  }
  c[gid] = sum;
}

__kernel void transpose(__global double *a, __global double *b, int width, int height) {
  int gid = get_global_id(0);
  int row = gid / width;
  int col = gid % width;
  b[col * height + row] = a[gid];
}

__kernel void add_value(__global double *a, __global double *b, double value) {
  int gid = get_global_id(0);
  b[gid] = a[gid] + value;
}

__kernel void mul_value(__global double *a, __global double *b, double value) {
  int gid = get_global_id(0);
  b[gid] = a[gid] * value;
}

__kernel void div_value(__global double *a, __global double *b, double value) {
  int gid = get_global_id(0);
  b[gid] = a[gid] / value;
}

"#;

const KERNEL_MATRIX_ADD_NAME: &str = "add";
const KERNEL_MATRIX_ADD_VALUE_NAME: &str = "add_value";
const KERNEL_MATRIX_SUB_NAME: &str = "sub";
const KERNEL_MATRIX_DIV_NAME: &str = "div";
const KERNEL_MATRIX_DIV_VALUE_NAME: &str = "div_value";
const KERNEL_MATRIX_MUL_NAME: &str = "mul";
const KERNEL_MATRIX_MUL_VALUE_NAME: &str = "mul_value";
const KERNEL_MATRIX_MUL_WISE_NAME: &str = "mul_wise";
const KERNEL_MATRIX_TRANSPOSE_NAME: &str = "transpose";

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

    if let Ok(device_id) = get_all_devices(CL_DEVICE_TYPE_GPU){
      let d = Device::new(device_id.first().unwrap().clone());

      Neuron::logger().info(|| format!("OpenCL device (MatrixMathOCL): {}", d.name().unwrap()));

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
}

impl MatrixMathExecutor for MatrixMathOCL {  
  fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
      // Check that the tensors are the same size
      assert!(a.rows == b.rows && a.cols == b.cols);

      // Create a new tensor to store the result
      let mut result = Tensor::zeros(a.rows, a.cols);

      if let Some(ref context) = self.context {
        if let Some(ref queue) = self.queue {
          if let Some(ref program) = self.program {
            let a_ocl = a.get_ocl_buffer();
            let b_ocl = b.get_ocl_buffer();
  
            let ab = a_ocl.lock().unwrap();
            let bb = b_ocl.lock().unwrap();
  
            let rb = unsafe {
              Buffer::<cl_double>::create(context, CL_MEM_WRITE_ONLY, result.data().len(), ptr::null_mut()).unwrap()
            };  

            let kernel = Kernel::create(&program, KERNEL_MATRIX_ADD_NAME).unwrap();

            let width: cl_int = result.cols as i32;

            let kernel_event = unsafe {
              ExecuteKernel::new(&kernel)
                  .set_arg(&*ab)
                  .set_arg(&*bb)
                  .set_arg(&rb)
                  .set_arg(&width)
                  .set_global_work_size(result.data().len())
                  .enqueue_nd_range(&queue).unwrap()
            };

            let mut events: Vec<cl_event> = Vec::default();
            events.push(kernel_event.get());
            
            let ret = unsafe { queue.enqueue_read_buffer(&rb, CL_NON_BLOCKING, 0, &mut result.data, &events).unwrap() };
            let error = ret.wait();

            if let Err(error) = error {
              Neuron::logger().error(|| format!("OpenCL Error: {:?}", error));
              std::process::exit(0);
            }  
          }
        }
      }

      Neuron::logger().debug(|| format!("OpenCL add matrix = {:?}", result));
      result
  }

  fn sub(&self, a: &Tensor, b: &Tensor) -> Tensor {
    // Check that the tensors are the same size
    assert!(a.rows == b.rows && a.cols == b.cols);

    // Create a new tensor to store the result
    let mut result = Tensor::zeros(a.rows, a.cols);

    if let Some(ref context) = self.context {
      if let Some(ref queue) = self.queue {
        if let Some(ref program) = self.program {
          let a_ocl = a.get_ocl_buffer();
          let b_ocl = b.get_ocl_buffer();

          let ab = a_ocl.lock().unwrap();
          let bb = b_ocl.lock().unwrap();

          let rb = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_WRITE_ONLY, result.data().len(), ptr::null_mut()).unwrap()
          };  

          let kernel = Kernel::create(&program, KERNEL_MATRIX_SUB_NAME).unwrap();

          let width: cl_int = result.cols as i32;

          let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&*bb)
                .set_arg(&rb)
                .set_arg(&width)
                .set_global_work_size(result.data().len())
                .enqueue_nd_range(&queue).unwrap()
          };

          let mut events: Vec<cl_event> = Vec::default();
          events.push(kernel_event.get());
          
          let ret = unsafe { queue.enqueue_read_buffer(&rb, CL_NON_BLOCKING, 0, &mut result.data, &events).unwrap() };
          let error = ret.wait();

          if let Err(error) = error {
            Neuron::logger().error(|| format!("OpenCL Error: {:?}", error));
            std::process::exit(0);
          }  
        }
      }
    }

    Neuron::logger().debug(|| format!("OpenCL add matrix = {:?}", result));
    result
  }

  fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
    // Check that the tensors are compatible for multiplication
    assert!(a.cols == b.rows);

    let timer = Instant::now();

    // Create a new tensor to store the result
    let mut result = Tensor::zeros(a.rows, b.cols);

    Neuron::logger().profiling(|| format!("Matrix Mul (results created) = {}ms", timer.elapsed().as_millis()));

    if let Some(ref context) = self.context {
      if let Some(ref queue) = self.queue {
        if let Some(ref program) = self.program {
          let a_ocl = a.get_ocl_buffer();
          let b_ocl = b.get_ocl_buffer();

          let ab = a_ocl.lock().unwrap();
          let bb = b_ocl.lock().unwrap();
          let rb = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_WRITE_ONLY, result.data().len(), ptr::null_mut()).unwrap()
          };  

          Neuron::logger().profiling(|| format!("Matrix Mul (opencl memory) = {}ms", timer.elapsed().as_millis()));

          let kernel = Kernel::create(&program, KERNEL_MATRIX_MUL_NAME).unwrap();

          Neuron::logger().profiling(|| format!("Matrix Mul (opencl kernel created) = {}ms", timer.elapsed().as_millis()));

          let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&*bb)
                .set_arg(&rb)
                .set_arg(&(a.cols as i32))
                .set_arg(&(b.cols as i32))
                .set_arg(&(result.cols as i32))
                .set_global_work_size(result.data().len())
                .enqueue_nd_range(&queue).unwrap()
          };

          let mut events: Vec<cl_event> = Vec::default();
          events.push(kernel_event.get());
          
          Neuron::logger().profiling(|| format!("Matrix Mul (kernel executed) = {}ms", timer.elapsed().as_millis()));

          let ret = unsafe { queue.enqueue_read_buffer(&rb, CL_NON_BLOCKING, 0, &mut result.data, &events).unwrap() };
          let error = ret.wait();

          if let Err(error) = error {
            Neuron::logger().error(|| format!("OpenCL Error: {:?}", error));
            std::process::exit(0);
          }

          Neuron::logger().profiling(|| format!("Matrix Mul (opencl result readed) = {}ms", timer.elapsed().as_millis()));

        }
      }
    }

    Neuron::logger().debug(|| format!("OpenCL multiply matrix = {:?}", result));
    result
  }

  fn mul_wise(&self, a: &Tensor, b: &Tensor) -> Tensor {
    // Check that the tensors are compatible for multiplication
    assert!(a.rows == b.rows && a.cols == b.cols);

    // Create a new tensor to store the result
    let mut result = Tensor::zeros(a.rows, a.cols);

    if let Some(ref context) = self.context {
      if let Some(ref queue) = self.queue {
        if let Some(ref program) = self.program {
          let a_ocl = a.get_ocl_buffer();
          let b_ocl = b.get_ocl_buffer();

          let ab = a_ocl.lock().unwrap();
          let bb = b_ocl.lock().unwrap();
          let rb = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_WRITE_ONLY, result.data().len(), ptr::null_mut()).unwrap()
          };  

          let kernel = Kernel::create(&program, KERNEL_MATRIX_MUL_WISE_NAME).unwrap();

          let width: cl_int = result.cols as i32;

          let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&*bb)
                .set_arg(&rb)
                .set_arg(&width)
                .set_global_work_size(result.data().len())
                .enqueue_nd_range(&queue).unwrap()
          };

          let mut events: Vec<cl_event> = Vec::default();
          events.push(kernel_event.get());
          
          let ret = unsafe { queue.enqueue_read_buffer(&rb, CL_NON_BLOCKING, 0, &mut result.data, &events).unwrap() };
          let error = ret.wait();

          if let Err(error) = error {
            Neuron::logger().error(|| format!("OpenCL Error: {:?}", error));
            std::process::exit(0);
          }  
        }
      }
    }

    Neuron::logger().debug(|| format!("OpenCL multiply wise matrix = {:?}", result));
    result
  }

  fn div(&self, a: &Tensor, b: &Tensor) -> Tensor {
      // Check that the tensors are the same size
      assert!(a.rows == b.rows && a.cols == b.cols);

      // Create a new tensor to store the result
      let mut result = Tensor::zeros(a.rows, a.cols);

      if let Some(ref context) = self.context {
        if let Some(ref queue) = self.queue {
          if let Some(ref program) = self.program {
            let a_ocl = a.get_ocl_buffer();
            let b_ocl = b.get_ocl_buffer();
  
            let ab = a_ocl.lock().unwrap();
            let bb = b_ocl.lock().unwrap();
              let rb = unsafe {
              Buffer::<cl_double>::create(context, CL_MEM_WRITE_ONLY, result.data().len(), ptr::null_mut()).unwrap()
            };  

            let kernel = Kernel::create(&program, KERNEL_MATRIX_DIV_NAME).unwrap();

            let width: cl_int = result.cols as i32;

            let kernel_event = unsafe {
              ExecuteKernel::new(&kernel)
                  .set_arg(&*ab)
                  .set_arg(&*bb)
                  .set_arg(&rb)
                  .set_arg(&width)
                  .set_global_work_size(result.data().len())
                  .enqueue_nd_range(&queue).unwrap()
            };

            let mut events: Vec<cl_event> = Vec::default();
            events.push(kernel_event.get());
            
            let ret = unsafe { queue.enqueue_read_buffer(&rb, CL_NON_BLOCKING, 0, &mut result.data, &events).unwrap() };
            let error = ret.wait();

            if let Err(error) = error {
              Neuron::logger().error(|| format!("OpenCL Error: {:?}", error));
              std::process::exit(0);
            }  
          }
        }
      }
      Neuron::logger().debug(|| format!("OpenCL div matrix = {:?}", result));
      result
  }

  fn dot(&self, a: &Tensor, b: &Tensor) -> f64 {
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

  fn transpose(&self, a: &Tensor) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::zeros(a.cols, a.rows);

    if let Some(ref context) = self.context {
      if let Some(ref queue) = self.queue {
        if let Some(ref program) = self.program {
          let a_ocl = a.get_ocl_buffer();
          let ab = a_ocl.lock().unwrap();
          let rb = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_WRITE_ONLY, result.data().len(), ptr::null_mut()).unwrap()
          };  

          let kernel = Kernel::create(&program, KERNEL_MATRIX_TRANSPOSE_NAME).unwrap();

          let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&rb)
                .set_arg(&(a.cols as i32))
                .set_arg(&(a.rows as i32))
                .set_global_work_size(a.data().len())
                .enqueue_nd_range(&queue).unwrap()
          };

          let mut events: Vec<cl_event> = Vec::default();
          events.push(kernel_event.get());
          
          let ret = unsafe { queue.enqueue_read_buffer(&rb, CL_NON_BLOCKING, 0, &mut result.data, &events).unwrap() };
          let error = ret.wait();

          if let Err(error) = error {
            Neuron::logger().error(|| format!("OpenCL Error: {:?}", error));
            std::process::exit(0);
          }  
        }
      }
    }

    Neuron::logger().debug(|| format!("OpenCL transpose matrix = {:?}", result));
    result
  }

  fn add_value(&self, a: &Tensor, value: f64) -> Tensor {
    let mut result = Tensor::zeros(a.rows, a.cols);

    if let Some(ref context) = self.context {
      if let Some(ref queue) = self.queue {
        if let Some(ref program) = self.program {
          let a_ocl = a.get_ocl_buffer();
          let ab = a_ocl.lock().unwrap();
          let rb = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_WRITE_ONLY, result.data().len(), ptr::null_mut()).unwrap()
          };  

          let kernel = Kernel::create(&program, KERNEL_MATRIX_ADD_VALUE_NAME).unwrap();

          let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&rb)
                .set_arg(&value)
                .set_global_work_size(a.data().len())
                .enqueue_nd_range(&queue).unwrap()
          };

          let mut events: Vec<cl_event> = Vec::default();
          events.push(kernel_event.get());
          
          let ret = unsafe { queue.enqueue_read_buffer(&rb, CL_NON_BLOCKING, 0, &mut result.data, &events).unwrap() };
          let error = ret.wait();

          if let Err(error) = error {
            Neuron::logger().error(|| format!("OpenCL Error: {:?}", error));
            std::process::exit(0);
          }  
        }
      }
    }
    Neuron::logger().debug(|| format!("OpenCL add value matrix = {:?}", result));
    result
  }

  fn div_value(&self, a: &Tensor, value: f64) -> Tensor {
    let mut result = Tensor::zeros(a.rows, a.cols);

    if let Some(ref context) = self.context {
      if let Some(ref queue) = self.queue {
        if let Some(ref program) = self.program {
          let a_ocl = a.get_ocl_buffer();
          let ab = a_ocl.lock().unwrap();
          let rb = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_WRITE_ONLY, result.data().len(), ptr::null_mut()).unwrap()
          };  

          let kernel = Kernel::create(&program, KERNEL_MATRIX_DIV_VALUE_NAME).unwrap();

          let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&rb)
                .set_arg(&value)
                .set_global_work_size(a.data().len())
                .enqueue_nd_range(&queue).unwrap()
          };

          let mut events: Vec<cl_event> = Vec::default();
          events.push(kernel_event.get());
          
          let ret = unsafe { queue.enqueue_read_buffer(&rb, CL_NON_BLOCKING, 0, &mut result.data, &events).unwrap() };
          let error = ret.wait();

          if let Err(error) = error {
            Neuron::logger().error(|| format!("OpenCL Error: {:?}", error));
            std::process::exit(0);
          }  
        }
      }
    }

    Neuron::logger().debug(|| format!("OpenCL div value matrix = {:?}", result));

    result
  }

  fn mul_value(&self, a: &Tensor, value: f64) -> Tensor {
    let mut result = Tensor::zeros(a.rows, a.cols);

    if let Some(ref context) = self.context {
      if let Some(ref queue) = self.queue {
        if let Some(ref program) = self.program {
          let a_ocl = a.get_ocl_buffer();
          let ab = a_ocl.lock().unwrap();
          let rb = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_WRITE_ONLY, result.data().len(), ptr::null_mut()).unwrap()
          };  

          let kernel = Kernel::create(&program, KERNEL_MATRIX_MUL_VALUE_NAME).unwrap();

          let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&*ab)
                .set_arg(&rb)
                .set_arg(&value)
                .set_global_work_size(a.data().len())
                .enqueue_nd_range(&queue).unwrap()
          };

          let mut events: Vec<cl_event> = Vec::default();
          events.push(kernel_event.get());
          
          let ret = unsafe { queue.enqueue_read_buffer(&rb, CL_NON_BLOCKING, 0, &mut result.data, &events).unwrap() };
          let error = ret.wait();

          if let Err(error) = error {
            Neuron::logger().error(|| format!("OpenCL Error: {:?}", error));
            std::process::exit(0);
          }  
        }
      }
    }

    Neuron::logger().debug(|| format!("OpenCL mul value matrix = {:?}", result));

    result
  }

  fn sum_row(&self, a:&Tensor) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::zeros(a.rows, 1);

    for i in 0..a.rows {
      let mut sum: f64 = 0.0;
      for j in 0..a.cols {
        sum += a.get(i, j);
      }
      result.set(i, 0, sum);
    }

    result
  }

  fn broadcast(&self, a: &Tensor, rows: usize, cols: usize) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::zeros(rows, cols);

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

    result
  }

  fn pad(&self, a: &Tensor, pad_row: usize, pad_col: usize) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::zeros(a.rows() + pad_row*2, a.cols() + pad_col*2);

    //println!("pad input = {}", a);

    for i in 0..a.rows() {
      for j in 0..a.cols() {
          result.set(i+pad_row, j+pad_col,a.get(i,j));
      }
    }

    //println!("pad output = {}", result);

    result
  }


}

#[derive(Debug)]
pub struct TensorOCL {
  buffer: Arc<Mutex<Buffer<cl_double>>>,
}

impl TensorOCL {
  pub fn new(buffer: &Vec<f64>) -> Option<TensorOCL> {

    let executor = Neuron::matrix();
    if let MatrixMathExecutorEnum::OCL(ref matrix_ocl) = **executor {
      let mut ocl_buffer = unsafe {
        Buffer::<cl_double>::create(matrix_ocl.get_ocl_context().unwrap(), CL_MEM_READ_ONLY, buffer.len(), ptr::null_mut()).unwrap()
      };
  
      let write_event = unsafe { matrix_ocl.get_ocl_queue().unwrap().enqueue_write_buffer(&mut ocl_buffer, CL_BLOCKING, 0, buffer, &[]).unwrap() };
      let error = write_event.wait();
  
      if let Err(error) = error {
        Neuron::logger().error(|| format!("OpenCL Error: {:?}", error));
        std::process::exit(0);
      }  
  
      return Some(TensorOCL {
        buffer: Arc::new(Mutex::new(ocl_buffer)),
      });
    }
    None 
  }
}

pub trait OCL {
  fn get_ocl_buffer(&self) -> Arc<Mutex<Buffer<cl_double>>>;
}

impl OCL for Tensor {
  fn get_ocl_buffer(&self) -> Arc<Mutex<Buffer<cl_double>>> {
      let r = self.tensor_ocl.as_ref().unwrap().clone();
      let r1 = r.lock().unwrap();
      r1.buffer.clone()
  }
}