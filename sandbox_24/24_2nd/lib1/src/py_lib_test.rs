#![allow(dead_code)]

#[derive(Debug, Clone)]
enum Config {
    Default,
    LinearAnn(usize, usize, f32, f32),
}

impl Config {
    
}

#[cfg(test)]
mod test {
    use super::*;
    
    #[test]
    fn test1() {
        let config = Config::LinearAnn(0, 200, 0.0, 1.0);
        println!("{:?}", config);
    }
}
