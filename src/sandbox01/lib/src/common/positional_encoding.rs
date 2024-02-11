use tch::Tensor;

/// # About
///
/// (バッチサイズ, トークン数, 埋め込み次元)のTensorを受け取り、位置埋め込みベクトルを足したTensorを返すクラスです.
///
/// 位置埋め込みベクトルは、トークン数が使用中変化しないことを前提とし、(トークン数, 埋め込み次元)の形状のTensorをキャッシュとして保持します.
pub struct PositionalEncoding {
    /// (トークン数, 埋め込み次元)の形状の[`Tensor`].
    positional_encoding: Tensor,
}

impl PositionalEncoding {
    /// # About
    ///
    /// コンストラクタ.
    ///
    /// ## Detail
    ///
    /// コンストラクタ時に、位置埋め込みベクトルを作成します.
    pub fn new(pos_max: i64, d_model: i64, device: tch::Device) -> Self {
        let mut positional_encoding = vec![];
        for i in 0..pos_max {
            // (d_model)
            let pos_tensor = Self::single_positonal_encoding(i, d_model);
            positional_encoding.push(pos_tensor);
        }
        // (pos_max, d_model)
        let positional_encoding = Tensor::stack(&positional_encoding, 0).to_kind(tch::Kind::Float).to_device(device);

        Self { positional_encoding }
    }

    /// # About
    /// 
    /// (バッチサイズ, トークン数, 埋め込み次元)の形状のTensorを受け取り、位置埋め込みベクトルを足したTensorを返します.
    /// 
    /// ## Args
    /// 
    /// - `x: Tensor` : (バッチサイズ, トークン数, 埋め込み次元)の形状のTensor.
    /// 
    /// ## Return
    /// 
    /// - `Tensor`: 位置埋め込みベクトルを足したTensor. (バッチサイズ, トークン数, 埋め込み次元)
    pub fn positional_encoding(&self, x: Tensor) -> Tensor {
        let batch_size = x.size()[0];
        let pos_enc = (0..batch_size).map(|_| self.positional_encoding.copy()).collect::<Vec<_>>();
        let pos_enc = Tensor::stack(&pos_enc, 0);

        assert_eq!(x.size(), pos_enc.size());
        let x = x + pos_enc;
        x
    }

    /// (d_model)の形状のTensorを返します.
    fn single_positonal_encoding(pos: i64, d_model: i64) -> Tensor {
        let pos = pos as f32;
        let d_model = d_model as f32;

        let mut x = vec![];
        for i in 0..d_model as usize {
            let ii = ((i / 2) * 2) as f32;
            let w = pos / (10000_f32.powf(ii / d_model));
            if i % 2 == 0 {
                x.push(w.sin());
            } else {
                x.push(w.cos());
            }
        }
        Tensor::of_slice(x.as_slice()).to_kind(tch::Kind::Float)
    }
}
