use tch::Tensor;

pub struct DefaultRewardNonLinear1 {
    // 設定値
    /// 変換開始ステップ
    start_step: usize,

    /// 変換終了ステップ
    end_step: usize,

    /// 変換前情報計算用の移動平均幅
    n_move_ave: usize,

    // 内部情報.
    /// カウンタ.
    count_opt: usize,

    /// 温度パラメータ.
    p: f32,

    /// 温度パラメータデルタ.
    dp: f32,

    /// リングバッファ.
    r_cache: Vec<f32>,

    /// リングバッファ用のインデックスカウンタ.
    i_r_cache: usize,
}

impl DefaultRewardNonLinear1 {
    pub fn new(start_step: usize, end_step: usize, start_p: f32, end_p: f32, n_move_ave: usize) -> Self {
        let dp = (end_p - start_p) / (end_step - start_step) as f32;
        Self {
            start_step,
            end_step,
            n_move_ave,
            count_opt: 0,
            p: start_p,
            dp,
            r_cache: vec![],
            i_r_cache: 0,
        }
    }

    pub fn reward(&mut self, reward: Tensor) -> Tensor {
        // 呼び出し回数カウント.
        self.count_opt += 1;

        // 変換開始までは事前情報の計算だけ行う.
        if self.count_opt < self.start_step {
            // 報酬の移動平均ようのバッファへの追加処理.
            let r = Vec::<f32>::try_from(&reward).expect("failed tensor to vec");
            let r_ave = r.iter().sum::<f32>() / r.len() as f32;
            self.push_r_cache(r_ave);

            // 報酬自体はそのまま返す.
            reward
        }
        // 変換開始から終了までは、事前情報を固定しながら、温度パラメータを変化させながら変換を行う.
        else if self.count_opt < self.end_step {
            // 報酬変換用の情報取得.
            let bottom = self.get_ave_r_cache();
            let top = 1.0;

            // 報酬の変換処理.
            let reward = self.reward_function(reward, bottom, top, self.p);
            
            // rewardリターン前に温度パラメータを更新.
            self.p += self.dp;

            // return
            reward
        }
        // 変換終了以降は、事前情報と温度パラメータ両方を固定して変換を行う.
        else {
            // 報酬変換用の情報取得.
            let bottom = self.get_ave_r_cache();
            let top = 1.0;

            // 報酬の変換処理.
            let reward = self.reward_function(reward, bottom, top, self.p);

            // 温度パラメータは更新しない.
            // return
            reward
        }
    }

    fn reward_function(&self, reward: Tensor, bottom: f32, _top: f32, p: f32) -> Tensor {
        let reward_vec = Vec::<f32>::try_from(&reward).expect("failed tensor to vec");
        let alpha = bottom;
        let beta = bottom / p;

        let b = alpha * (beta - 1.0) / (alpha - beta);

        let mut r = vec![];
        for x in reward_vec.iter() {
            let y = - b * (1.0 + b) / (x + b) + 1.0 + b;
            r.push(y);
        }

        let r = Tensor::of_slice(&r).to_kind(tch::Kind::Float).to_device(tch::Device::Cpu);
        r
    }

    fn push_r_cache(&mut self, r: f32) {
        // 一週目.
        if self.r_cache.len() <= self.n_move_ave {
            self.r_cache.push(r);
        }
        // 二週目以降のリングバッファ処理.
        else {
            self.r_cache[self.i_r_cache] = r;
            self.i_r_cache = (self.i_r_cache + 1) % self.n_move_ave;
        }
    }

    fn get_ave_r_cache(&self) -> f32 {
        self.r_cache.iter().sum::<f32>() / self.r_cache.len() as f32
    }
}


#[cfg(test)]
mod test {
    use tch;
    use super::*;

    #[test]
    fn tensor_test() {
        let t = tch::Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0]).to_kind(tch::Kind::Float).to_device(tch::Device::Cpu);
        t.print();
    }

    #[test]
    fn test_reward_func() {
        let drnl = DefaultRewardNonLinear1::new(0, 0, 3.0, 3.0, 1);
        let r = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0];
        let r = tch::Tensor::of_slice(&r).to_kind(tch::Kind::Float).to_device(tch::Device::Cpu);

        let r = drnl.reward_function(r, 0.75, 1.0, 3.0);
        r.print();
    }
}
