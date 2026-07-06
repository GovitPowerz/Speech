//! Neural network: layers, BLSTM, generic container, activations (Phase 2
//! forward + Phase 3 backward/BPTT and iRPROP- training; LSTM + dense scope).
//!
//! Ported from legacy C++: NeuralNetwork.hpp, BLSTMNeuralNetwork.*, LSTMLayer.*,
//! NeuronLayer.*, ActivationFunctions.h, Rprop.*/Trainer.h (train.rs).
//! SRN/CWRNN are dead code in the legacy (never ported); the conv layer is
//! broken-as-committed there (see IMPROVEMENTS.md).

pub mod activations;
pub mod blstm;
pub mod layers;
pub mod network;
pub mod train;
