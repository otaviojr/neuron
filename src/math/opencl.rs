use opencl3::{memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY}, context::Context, kernel::{Kernel, ExecuteKernel}, device::{Device, get_all_devices, CL_DEVICE_TYPE_GPU}, command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, program::Program, types::{cl_double, CL_BLOCKING, CL_NON_BLOCKING, cl_ulong, cl_event, cl_int}};
use std::{ptr};

use super::{MatrixMath, Tensor};

const PROGRAM_SOURCE: &str = r#"
__kernel void add(__global double *a, __global double *b, __global double *c, int width) {
  int gid = get_global_id(0);
  int row = gid / width;
  int col = gid % width;
  c[gid] = a[gid] + b[gid];
}

__kernel void multiply(__global double *a, __global double *b, __global double *c, int width) {
  int gid = get_global_id(0);
  int row = gid / width;
  int col = gid % width;
  double sum = 0.0;
  for (int i = 0; i < width; i++) {
      sum += a[row * width + i] * b[i * width + col];
  }
  c[gid] = sum;
}
"#;

const KERNEL_MATRIX_ADD_NAME: &str = "add";
const KERNEL_MATRIX_MULTIPLY_NAME: &str = "multiply";

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

    MatrixMathOCL {
      device,
      context,
      queue,
      program,
    }
  }
}

impl MatrixMath for MatrixMathOCL {
  fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
      // Check that the tensors are the same size
      assert!(a.rows == b.rows && a.cols == b.cols);

      // Create a new tensor to store the result
      let mut result = Tensor::zeros(a.rows, a.cols);

      if let Some(ref context) = self.context {
        if let Some(ref queue) = self.queue {
          if let Some(ref program) = self.program {
            let mut ab = unsafe {
              Buffer::<cl_double>::create(context, CL_MEM_READ_ONLY, a.data().len(), ptr::null_mut()).unwrap()
            };
            let mut bb = unsafe {
              Buffer::<cl_double>::create(context, CL_MEM_READ_ONLY, b.data().len(), ptr::null_mut()).unwrap()
            };
            let rb = unsafe {
              Buffer::<cl_double>::create(context, CL_MEM_WRITE_ONLY, result.data().len(), ptr::null_mut()).unwrap()
            };  

            let _ = unsafe { queue.enqueue_write_buffer(&mut ab, CL_BLOCKING, 0, a.data(), &[]).unwrap() };
            let write_event = unsafe { queue.enqueue_write_buffer(&mut bb, CL_NON_BLOCKING, 0, &b.data(), &[]).unwrap() };

            let kernel = Kernel::create(&program, KERNEL_MATRIX_ADD_NAME).unwrap();

            let width: cl_int = a.cols as i32;

            let kernel_event = unsafe {
              ExecuteKernel::new(&kernel)
                  .set_arg(&ab)
                  .set_arg(&bb)
                  .set_arg(&rb)
                  .set_arg(&width)
                  .set_global_work_size(result.data().len())
                  .set_wait_event(&write_event)
                  .enqueue_nd_range(&queue).unwrap()
            };

            let mut events: Vec<cl_event> = Vec::default();
            events.push(kernel_event.get());
            
            let ret = unsafe { queue.enqueue_read_buffer(&rb, CL_NON_BLOCKING, 0, &mut result.data, &events).unwrap() };
            let _ = ret.wait().unwrap();

          }
        }
      }
      println!("OpenCL add matrix = {:?}", result);
      result
  }

  fn sub(&self, a: &Tensor, b: &Tensor) -> Tensor {
      // Check that the tensors are the same size
      assert!(a.rows == b.rows && a.cols == b.cols);

      // Create a new tensor to store the result
      let mut result = Tensor::zeros(a.rows, a.cols);

      // Perform element-wise subtraction
      for i in 0..a.data.len() {
          result.data[i] = a.data[i] - b.data[i];
      }

      result
  }

  fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
    // Check that the tensors are compatible for multiplication
    assert!(a.cols == b.rows);

    // Create a new tensor to store the result
    let mut result = Tensor::zeros(a.rows, b.cols);

    if let Some(ref context) = self.context {
      if let Some(ref queue) = self.queue {
        if let Some(ref program) = self.program {
          let mut ab = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_READ_ONLY, a.data().len(), ptr::null_mut()).unwrap()
          };
          let mut bb = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_READ_ONLY, b.data().len(), ptr::null_mut()).unwrap()
          };
          let rb = unsafe {
            Buffer::<cl_double>::create(context, CL_MEM_WRITE_ONLY, result.data().len(), ptr::null_mut()).unwrap()
          };  

          let _ = unsafe { queue.enqueue_write_buffer(&mut ab, CL_BLOCKING, 0, a.data(), &[]).unwrap() };
          let write_event = unsafe { queue.enqueue_write_buffer(&mut bb, CL_NON_BLOCKING, 0, &b.data(), &[]).unwrap() };

          let kernel = Kernel::create(&program, KERNEL_MATRIX_MULTIPLY_NAME).unwrap();

          let width: cl_int = a.cols as i32;

          let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&ab)
                .set_arg(&bb)
                .set_arg(&rb)
                .set_arg(&width)
                .set_global_work_size(a.data().len())
                .set_wait_event(&write_event)
                .enqueue_nd_range(&queue).unwrap()
          };

          let mut events: Vec<cl_event> = Vec::default();
          events.push(kernel_event.get());
          
          let ret = unsafe { queue.enqueue_read_buffer(&rb, CL_NON_BLOCKING, 0, &mut result.data, &events).unwrap() };
          let error = ret.wait();

          if let Err(error) = error {
            println!("OpenCL Error: {:?}", error);
            std::process::exit(0);
          }
        }
      }
    }
    println!("OpenCL add matrix = {:?}", result);
    result
  }

  fn mul_wise(&self, a: &Tensor, b: &Tensor) -> Tensor {
    // Check that the tensors are compatible for multiplication
    assert!(a.rows == b.rows && a.cols == b.cols);

    let data: Vec<f64> = a.data()
    .iter()
    .zip(b.data().iter())
    .map(|(v1, v2)| v1 * v2).collect();

    
    Tensor::from_data(a.rows, a.cols, data)
  }

  fn div(&self, a: &Tensor, b: &Tensor) -> Tensor {
      // Check that the tensors are the same size
      assert!(a.rows == b.rows && a.cols == b.cols);

      // Create a new tensor to store the result
      let mut result = Tensor::zeros(a.rows, a.cols);

      // Perform element-wise division
      for i in 0..a.data.len() {
          result.data[i] = a.data[i] / b.data[i];
      }

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

    // Transpose the matrix
    for i in 0..a.rows {
        for j in 0..a.cols {
            result.set(j, i, a.get(i, j));
        }
    }

    result
  }

  fn add_value(&self, a: &Tensor, value: f64) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::zeros(a.rows, a.cols);

    for i in 0..a.rows {
      for j in 0..a.cols {
        let val = a.get(i, j);
        result.set(i, j, val + value);
      }
    }

    result
  }

  fn div_value(&self, a: &Tensor, value: f64) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::zeros(a.rows, a.cols);

    for i in 0..a.rows {
      for j in 0..a.cols {
        let val = a.get(i, j);
        result.set(i, j, val/value);
      }
    }

    result
  }

  fn mul_value(&self, a: &Tensor, value: f64) -> Tensor {
    // Create a new tensor to store the result
    let mut result = Tensor::zeros(a.rows, a.cols);

    for i in 0..a.rows {
      for j in 0..a.cols {
        let val = a.get(i, j);
        result.set(i, j, val*value);
      }
    }

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