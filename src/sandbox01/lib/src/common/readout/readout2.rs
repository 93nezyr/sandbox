use tch::nn::{conv1d, Conv1D, ConvConfig, ModuleT};
use tch::{Device, Tensor};

/// # About
///
/// (batch_size, n_tokens, n_features)のTensorを受け取り、(batch_size, n_features)のTensorに変換するクラスです.
///
/// ## Detalis
///
/// n_tokens次元方向に、フィルターサイズ(2)の畳み込みを繰り返し、(batch_size, 1, n_features)のTensorに変換した後、squeezeして、(batch_size, n_features)のTensorに変換します.
/// 
/// n_features次元方向の情報は、グループ畳み込みを行うので、互いに独立して扱われます.
///
/// n_tokens次元方向は、時系列に並んだ情報であることを想定しています.
pub struct ReadoutTokenGroupingResCNN1D {
    device: Device,

    cnns: Vec<Conv1D>,
}

impl ReadoutTokenGroupingResCNN1D {
    /// # About
    ///
    /// コンストラクタです.
    ///
    /// ## Args
    ///
    /// - `p: &tch::nn::Path` - モデルのパラメータを保存するためのPath(デバイス情報も含まれています).
    /// - `n_tokens: i64` - トークン数. [`Self::forward_t`]の入力Tensorの2次元目の次元数.
    /// - `n_features: i64` - 特徴量数. [`Self::forward_t`]の入力Tensorの3次元目の次元数.
    pub fn new(p: &tch::nn::Path, device: Device, n_tokens: i64, n_features: i64) -> Self {
        let mut cnns = vec![];
        for i in 0..(n_tokens - 1) {
            let c = ConvConfig {
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: n_features,
                bias: true,
                ws_init: tch::nn::Init::KaimingUniform,
                bs_init: tch::nn::Init::Const(0.),
            };
            let cnn = conv1d(
                &(p / format!("readout_conv1d_{}", i + 1)),
                n_features,
                n_features,
                2,
                c,
            );
            cnns.push(cnn);
        }

        ReadoutTokenGroupingResCNN1D { device, cnns }
    }

    /// # About
    ///
    /// Readoutします.
    ///
    /// ## Args
    ///
    /// - `x: Tensor` - (batch_size, n_tokens, n_features_in)のTensor
    ///
    /// ## Returns
    ///
    /// - `Tensor` - (batch_size, n_features_in)のTensor
    pub fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        // (batch_size, n_features_in, n_tokens)に分割.
        // Conv1Dにとって、n_tokensが画素数に対応する.
        let mut x = x.transpose(1, 2).to_kind(tch::Kind::Float).to_device(self.device);
        for cnn in self.cnns.iter() {
            x = cnn.forward_t(&x, train).leaky_relu();
        }
        // self.cnns通過後は、(batch_size, n_features_in, 1)になっている.
        // (batch_size, 1, n_features_in)に変換.
        x = x.transpose(1, 2).to_kind(tch::Kind::Float).to_device(self.device);
        // squeezeして、(batch_size, n_features_in)に変換.
        x.squeeze().to_kind(tch::Kind::Float).to_device(self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Readout1の挙動を確認します.
    #[test]
    fn test_readout1() {
        let var_store = tch::nn::VarStore::new(tch::Device::Cpu);
        let p = &var_store.root(); // Cannot Path divide by &str. So, use &Path.
        let batch_size = 2;
        let n_tokens = 10;
        let n_features_in = 5;

        let in_tensor = Tensor::randn(
            &[batch_size, n_tokens, n_features_in],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        println!("in_tensor: {:?}", in_tensor.size());

        let readout1 = ReadoutTokenGroupingResCNN1D::new(p, tch::Device::Cpu, n_tokens, n_features_in);
        let out_tensor = readout1.forward_t(&in_tensor, true);
        println!("out_tensor: {:?}", out_tensor.size());
    }

    /// CNNの挙動を確認します.
    #[test]
    fn test_cnn() {
        let var_store = tch::nn::VarStore::new(tch::Device::Cpu);
        let p = &var_store.root(); // Cannot Path divide by &str. So, use &Path.
        let batch_size = 2;
        let n_tokens = 10;
        let n_features_in = 5;
        // let n_features_out = n_features_in;

        let in_tensor = Tensor::randn(
            &[batch_size, n_tokens, n_features_in],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        println!("in_tensor: {:?}", in_tensor.size());

        // let c = tch::nn::ConvConfig::default();
        // <https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d>
        // groupsをin_channelに設定しつつ、conv1dのin_channelとout_channelを同じにすることで、出力の各チャネルは、入力の対応する1チャネルからのみ情報を取得することになる.
        // groupsは、(おそらく)近傍nチャネルの情報を出力の各チャネルに取り込む設定で、groupsの値でin_channelとout_channelを割り切れる必要がある.
        let c = tch::nn::ConvConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: n_tokens,
            bias: true,
            ws_init: tch::nn::Init::KaimingUniform,
            bs_init: tch::nn::Init::Const(0.),
        };
        let cnn_1d_1 = tch::nn::conv1d(&(p / "cnn_1d_1"), n_tokens, n_tokens, 2, c);
        println!("cnn_1d_1.ws: {:?}", cnn_1d_1.ws.size());

        let out_tensor = cnn_1d_1.forward_t(&in_tensor, true);
        println!("out_tensor: {:?}", out_tensor.size());
    }

    /// Readout用のCNNの挙動を確認します.
    #[test]
    fn test_readout_cnn() {
        let var_store = tch::nn::VarStore::new(tch::Device::Cpu);
        let p = &var_store.root(); // Cannot Path divide by &str. So, use &Path.
        let batch_size = 2;
        let n_tokens = 3;
        let n_features_in = 5;

        // (batch_size, n_tokens, n_features_in)
        let in_tensor = Tensor::randn(
            &[batch_size, n_tokens, n_features_in],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        println!("in_tensor: {:?}", in_tensor.size());
        in_tensor.print();
        // (batch_size, n_features_in, n_tokens)
        let in_tensor = in_tensor.transpose(1, 2);
        println!("in_tensor: {:?}", in_tensor.size());
        in_tensor.print();

        let c = tch::nn::ConvConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: n_features_in,
            bias: true,
            ws_init: tch::nn::Init::KaimingUniform,
            bs_init: tch::nn::Init::Const(0.),
        };
        let cnn_1d_1 = tch::nn::conv1d(&(p / "cnn_1d_1"), n_features_in, n_features_in, 2, c);
        println!("cnn_1d_1.ws: {:?}", cnn_1d_1.ws.size());

        let out_tensor = cnn_1d_1.forward_t(&in_tensor, true);
        println!("out_tensor: {:?}", out_tensor.size());

        let out_tensor = out_tensor.transpose(1, 2).contiguous();
        println!("out_tensor: {:?}", out_tensor.size());
    }
}
