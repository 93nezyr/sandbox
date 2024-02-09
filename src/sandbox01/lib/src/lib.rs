mod pmodel;
mod common;

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
