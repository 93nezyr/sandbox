mod pmodel;
mod common;

pub use common::{positional_encoding::PositionalEncoding, transformer::TransformerEncoder};
pub use common::readout;

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_tensor() {
        let t = tch::Tensor::of_slice(&[1, 2, 3, 4, 5]);
        t.print();
        let t = t.view([1, 1, 1, 5]);
        t.print();
    }
}

#[cfg(test)]
#[allow(dead_code)]
mod sandbox {
    #[test]
    fn sandbox1() {
        println!("hoge");
        let mut v = vec![];
        for i in 0..100 {
            v.push(i);
        }
    }
}
