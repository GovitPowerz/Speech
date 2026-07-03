//! Neural network: layers, BLSTM, generic container, activations (Phase 2,
//! forward only; LSTM + dense scope) and training (Phase 3 stub).
//!
//! Ported from legacy C++: NeuralNetwork.hpp, BLSTMNeuralNetwork.*, LSTMLayer.*,
//! NeuronLayer.*, ActivationFunctions.h; Rprop.*/Trainer.h pending (Phase 3).
//! SRN/CWRNN are dead code in the legacy (never ported); the conv layer is
//! broken-as-committed there (see IMPROVEMENTS.md).

pub mod activations;
pub mod blstm;
pub mod layers;
pub mod network;
pub mod train;
