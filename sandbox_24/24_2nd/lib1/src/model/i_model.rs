pub trait Model {
    type Input;

    type Output;

    type Config;

    fn build(config: Self::Config, device: tch::Device) -> Self;

    fn forward(&mut self, input: Self::Input) -> Self::Output;

    fn backward(&mut self, target: tch::Tensor) -> f32;
}