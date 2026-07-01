//! Neural network: layers, BLSTM, generic container, activations, training.
//!
//! Ported from legacy C++: NeuralNetwork.hpp, BLSTMNeuralNetwork.*, *Layer.*,
//! ActivationFunctions.h, Rprop.*, Trainer.h.

pub mod activations;
pub mod blstm;
pub mod layers;
pub mod network;
pub mod train;
