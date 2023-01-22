use std::fmt;
use std::fs::File;
use std::io::{self, Write};

#[derive(PartialEq, PartialOrd, Debug)]
#[allow(unused_variables)]
pub enum LogLevel {
  Debug,
  Info,
  Warn,
  Error,
}

impl fmt::Display for LogLevel {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
      match *self {
          LogLevel::Error => write!(f, "Error"),
          LogLevel::Warn => write!(f, "Warn"),
          LogLevel::Info => write!(f, "Info"),
          LogLevel::Debug => write!(f, "Debug"),
      }
  }
}

pub struct Logger {
    log_level: LogLevel,
    file: Option<File>,
}

impl fmt::Display for Logger {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
      write!(f, "Logger(level: {:?}, file: {:?})", self.log_level, self.file)
  }
}

impl Logger {
    pub fn new(log_level: LogLevel) -> Logger {
        Logger {
            log_level,
            file: None,
        }
    }

    pub fn new_with_file(file: Option<File>, log_level: LogLevel) -> Logger {
        Logger {
            log_level,
            file,
        }
    }

    pub fn set_log_level(&mut self, log_level: LogLevel) {
        self.log_level = log_level;
    }

    pub fn set_file(&mut self, file: File) {
        self.file = Some(file);
    }

    pub fn log(&mut self, log_level: LogLevel, message: &str) {
        if log_level >= self.log_level {
            let output = format!("[{}] {}", log_level, message);
            match &mut self.file {
                Some(file) => writeln!(file, "{}", output).unwrap(),
                None => writeln!(io::stdout(), "{}", output).unwrap(),
            };
        }
    }

    pub fn debug(&mut self, message: &str) {
        self.log(LogLevel::Debug, message);
    }

    pub fn info(&mut self, message: &str) {
        self.log(LogLevel::Info, message);
    }

    pub fn error(&mut self, message: &str) {
        self.log(LogLevel::Error, message);
    }

    pub fn warn(&mut self, message: &str) {
        self.log(LogLevel::Warn, message);
    }
}