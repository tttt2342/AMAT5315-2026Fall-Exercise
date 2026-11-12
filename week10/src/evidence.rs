use anyhow::Result;
use serde_json::Value;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

pub fn write_f64(path: &Path, data: &[f64]) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    for value in data {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

pub fn write_json(path: &Path, data: &Value) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut file, data)?;
    writeln!(file)?;
    file.flush()?;
    Ok(())
}
