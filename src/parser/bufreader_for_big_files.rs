use std::{
    fs::File,
    io::{self, prelude::*},
    rc::Rc,
};

/// The BufReaderMl is a BufReader that reuses a String multiples times (once for each line)
/// We gain some performance using that approach
pub struct BufReaderMl {
    reader: io::BufReader<File>,
    buf: Rc<String>,
}

/// Creates a new String which will hold the data of each line in a file
fn new() -> Rc<String> {
    Rc::new(String::with_capacity(1024)) // Tweakable capacity
}

impl BufReaderMl {
    /// Opens a file with a BufReader
    pub fn open(path: impl AsRef<std::path::Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = io::BufReader::new(file);
        let buf = new();

        Ok(Self { reader, buf })
    }
}

/// Uses the declared String for each line and does not create a new String for each individual line
/// and increases the performance.
impl Iterator for BufReaderMl {
    type Item = io::Result<Rc<String>>;

    fn next(&mut self) -> Option<Self::Item> {
        let buf = match Rc::get_mut(&mut self.buf) {
            // if the String holds information -> discard it so there is room for the next line
            Some(buf) => {
                buf.clear();
                buf
            }
            None => {
                self.buf = new();
                Rc::make_mut(&mut self.buf)
            }
        };

        self.reader
            .read_line(buf)
            .map(|u| {
                if u == 0 {
                    None
                } else {
                    Some(Rc::clone(&self.buf))
                }
            })
            .transpose()
    }
}
