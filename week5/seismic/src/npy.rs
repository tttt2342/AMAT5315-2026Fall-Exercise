use std::{fs, io::Write, path::Path};
pub fn write(path: &Path, shape: &[usize], data: &[f64], single: bool) {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let dims = shape
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let mut h = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({dims},), }}",
        if single { "<f4" } else { "<f8" }
    );
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut f = std::io::BufWriter::new(fs::File::create(path).unwrap());
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(h.len() as u16).to_le_bytes()).unwrap();
    f.write_all(h.as_bytes()).unwrap();
    for x in data {
        if single {
            f.write_all(&(*x as f32).to_le_bytes()).unwrap()
        } else {
            f.write_all(&x.to_le_bytes()).unwrap()
        }
    }
}
pub fn read(path: &Path, shape: &[usize]) -> Vec<f64> {
    let b = fs::read(path).unwrap();
    assert_eq!(&b[..8], b"\x93NUMPY\x01\x00");
    let hlen = u16::from_le_bytes([b[8], b[9]]) as usize;
    let h = std::str::from_utf8(&b[10..10 + hlen]).unwrap();
    assert!(h.contains("'<f8'") && h.contains("False"));
    let dims = h
        .split("'shape':")
        .nth(1)
        .unwrap()
        .split('(')
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let dims: Vec<usize> = dims
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    assert_eq!(dims, shape);
    let d = &b[10 + hlen..];
    assert_eq!(d.len(), shape.iter().product::<usize>() * 8);
    d.chunks_exact(8)
        .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
