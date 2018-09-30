using NeuralNetwork.Helpers;
using System;

namespace NeuralNetwork.NetworkModels
{
	public class Synapse
	{
        private readonly Guid Id;
        private readonly Neuron InputNeuron;
        private readonly Neuron OutputNeuron;
        private double Weight;
        private double WeightDelta;

        public Synapse(Guid id, Neuron inputNeuron, Neuron outputNeuron, double weight, double weightDelta) {
            this.Id = id;
            this.InputNeuron = inputNeuron;
            this.OutputNeuron = outputNeuron;
            this.Weight = weight;
            this.WeightDelta = weightDelta;
        }
        public Synapse(Neuron inputNeuron, Neuron outputNeuron)
        {
            Id = Guid.NewGuid();
            InputNeuron = inputNeuron;
            OutputNeuron = outputNeuron;
            Weight = Network.GetRandom();
        }

        public HelperSynapse ToHelperSynapse()
        {            
            return new HelperSynapse(Id,InputNeuron.ToHelperNeuron().Id,OutputNeuron.ToHelperNeuron().Id,Weight,WeightDelta);
        }
        public double CalculateGradient()
        {
            return OutputNeuron.CalculateGradientByWeight(Weight);
        }
        public double CalculateValue()
        {
            return InputNeuron.CalculateValueByWeight(Weight);
        }
        internal void NeuronBiasUpdated(object sender, NeuronBiasUpdatedEventArg e)
        {
            var prevDelta = WeightDelta;
            WeightDelta =InputNeuron.CalculateWeightDelta( e.LearnRate ,  e.Gradient );
            Weight += WeightDelta + e.Momentum * prevDelta;
        }
    }
}