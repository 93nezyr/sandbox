use tch::Tensor;

pub struct DefaultRewardNonLinear2 {
    // 設定値
    /// 変換開始ステップ
    start_step: usize,

    /// 変換前情報計算用の移動平均幅
    n_move_ave: usize,

    // 内部情報.
    /// カウンタ.
    count_opt: usize,

    /// 報酬の急峻さを決める定数.
    /// 
    /// 大体、変化をとらえたいオーダーの逆数÷10程度の値を設定する.
    /// 
    /// 例：0.01オーダーをとらえたい場合
    /// 
    /// 1 / 0.01 = 100
    /// 
    /// 100 / 10 = 10
    /// 
    /// coef = 10
    coef: f32,

    /// リングバッファ.
    r_cache: Vec<f32>,

    /// リングバッファ用のインデックスカウンタ.
    i_r_cache: usize,
}

impl DefaultRewardNonLinear2 {
    pub fn new(start_step: usize, coef: f32, n_move_ave: usize) -> Self {
        Self {
            start_step,
            n_move_ave,
            count_opt: 0,
            coef,
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
        // 変換開始、bottomを固定したまま変換を行う.
        else {
            // 報酬変換用の情報取得.
            let bottom = self.get_ave_r_cache();

            // 報酬の変換処理.
            let reward = self.reward_function(reward, bottom);

            // 温度パラメータは更新しない.
            // return
            reward
        }
    }

    fn reward_function(&self, reward: Tensor, bottom: f32) -> Tensor {
        let device = reward.device();
        let reward_vec = Vec::<f32>::try_from(&reward).expect("failed tensor to vec");

        // let beta = 2.0 / bottom;
        let beta = self.coef;

        // println!("reward_vec: {:?}", reward_vec);

        let mut r = vec![];
        for x in reward_vec.iter() {
            let x = x - bottom;
            let x = x * beta;
            r.push(x);
        }

        // println!("r: {:?}", r);

        let r = Tensor::of_slice(&r).to_kind(tch::Kind::Float).to_device(device);
        let r = r.tanh() + 1.0;
        let r = r * 0.5;
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
        let drnl = DefaultRewardNonLinear2::new(0, 100.0, 1);
        // Vec生成.
        let mut r = vec![];
        let start = 0.74;
        let end = 0.76;
        let size = 10;
        let d = (end - start) / size as f32;
        for i in 0..size {
            r.push(start + d * (i + 1) as f32);
        }
        println!("r: {:?}", r);
        let r = tch::Tensor::of_slice(&r).to_kind(tch::Kind::Float).to_device(tch::Device::Cpu);

        let r = drnl.reward_function(r, 0.75);
        let o_r_vec = Vec::<f32>::try_from(&r).expect("failed tensor to vec");
        println!("o_r_vec: {:?}", o_r_vec);
    }

    #[test]
    fn test_tanh_func() {
        // Vec生成.
        let mut r = vec![];
        let start = -10.0;
        let end = 10.0;
        let size = 10;
        let d = (end - start) / size as f32;
        for i in 0..size {
            r.push(start + d * i as f32);
        }
        println!("r: {:?}", r);
        let r = tch::Tensor::of_slice(&r).to_kind(tch::Kind::Float).to_device(tch::Device::Cpu);
        // r.print();
        let r = r.tanh();
        let o_r_vec = Vec::<f32>::try_from(&r).expect("failed tensor to vec");
        println!("o_r_vec: {:?}", o_r_vec);

        let v = vec![-0.1, 0.0, 0.1];
        let t = Tensor::of_slice(&v).to_kind(tch::Kind::Float).to_device(tch::Device::Cpu);
        let t = t.tanh() + 1.0;
        let t = t * 0.5;
        t.print();

        let v = vec![-2.0, 0.0, 2.0];
        let t = Tensor::of_slice(&v).to_kind(tch::Kind::Float).to_device(tch::Device::Cpu);
        let t = t.tanh() + 1.0;
        let t = t * 0.5;
        t.print();
    }
}
