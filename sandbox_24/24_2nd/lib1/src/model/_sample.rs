use tch::nn::{Module, OptimizerConfig};
use tch::{nn, Device, Tensor};

const IN_DIM: i64 = 7;
const HIDDEN_NODES: i64 = 128;
const OUT_DIM: i64 = 2;

fn net(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            IN_DIM,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, HIDDEN_NODES, OUT_DIM, Default::default()))
}

pub fn sample_code_neural_network_train() {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = net(&vs.root());
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    for _idx in 1..500 {
        // Dummy mini-batches made of zeros.
        let xs = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).to_device(Device::Cpu).to_kind(tch::Kind::Float);
        let ys = Tensor::of_slice(&[3.5, 7.0]).to_device(tch::Device::Cpu).to_kind(tch::Kind::Float);
        let loss = my_module.forward(&xs).mse_loss(&ys, tch::Reduction::Mean);
        println!("loss: {:?}", loss);
        opt.backward_step(&loss);
    }
}
