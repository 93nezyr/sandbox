use super::Model34Config;
use super::super::Model;
use tch::{
    nn,
    nn::{Adam, ModuleT, Optimizer, OptimizerConfig},
    Device, Tensor,
};

/// # About
/// 
/// `Model34`は、[batch_size, dim_obs]の形状の`Tensor`を入力として受け取り、[batch_size, action_space, n_quantile]の形状の`Tensor`を出力するモデルです.
/// 
/// 分位点回帰を想定したニューラルネットワークを保持しています.
pub struct Model34 {
    /// ニューラルネットワークパラメータ.
    vs: nn::VarStore,

    /// オプティマイザー.
    opt: Optimizer<Adam>,


}

impl Model for Model34 {
    type Input = Tensor;

    type Output = Tensor;

    type Config = Model34Config;

    fn build(config: Model34Config, device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let opt = nn::Adam::default()
            .build(&vs, 0.0001)
            .expect("failed to build Adam optimizer");

        Self { vs, opt }
    }

    fn forward(&mut self, input: Self::Input) -> Self::Output {
        todo!()
    }

    fn backward(&mut self, loss: tch::Tensor) -> f32 {
        todo!()
    }
}
